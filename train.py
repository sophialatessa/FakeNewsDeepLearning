#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import sys
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
#import yaml
import math
import pickle

# Parameters
# ==================================================


# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .05, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("experiment", "all", "All subjects (all), Trump or Email.")

# Model Hyperparameters
tf.flags.DEFINE_boolean("enable_word_embeddings", True, "Enable/disable the word embedding (default: True)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_float("decay_coefficient", 2.5, "Decay coefficient (default: 2.5)")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

cfg = {'word_embeddings': {'default': 'word2vec', 'word2vec':
    {'path': os.getcwd() +'/GoogleNews-vectors-negative300.bin',
     'dimension': 300, 'binary': True}, 'glove': {'path': '../../data/glove.6B.100d.txt', 'dimension': 100, 'length': 400000}},
     'datasets': {'default': '20newsgroup', 'mrpolarity': {'positive_data_file': {'path': 'data/rt-polaritydata/rt-polarity.pos',
                                                                                  'info': 'Data source for the positive data'},
                                                           'negative_data_file': {'path': 'data/rt-polaritydata/rt-polarity.neg',
                                                                                  'info': 'Data source for the negative data'}},
                  '20newsgroup': {'categories': ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian'], 'shuffle': True, 'random_state': 42},
                  'localdata': {'container_path': '../../data/input/SentenceCorpus', 'categories': None, 'shuffle': True, 'random_state': 42}}}

dataset_name = cfg["datasets"]["default"]
if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
    embedding_name = cfg['word_embeddings']['default']
    embedding_dimension = cfg['word_embeddings'][embedding_name]['dimension']
else:
    embedding_dimension = FLAGS.embedding_dim

# Data Preparation
# ==================================================

positive_data_file = os.getcwd() + '/data/clean/real.pkl'
negative_data_file = os.getcwd() + '/data/clean/fake.pkl'

# Load and clean data
print("Loading data...")
import re
import string
def clean(text):
    text = re.sub(r'\([^)]*\)', '', text)
    text = ' '.join([s for s in text.split() if not any([c.isdigit() for c in s])])
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    text = ' '.join([s for s in text.split() if not any([not c.isalpha() for c in s])])
    text = re.sub(' +', ' ', text)
    text = text.lower()
    return text

# Build vocabulary
x_origin, y = data_helpers.load_data_and_labels(positive_data_file, negative_data_file)

x_text = [" ".join(clean(x).split(" ")[:1000]) for x in x_origin]

max_document_length = max([len(x.split(" ")) for x in x_text])
print(max_document_length, "is mdl (max document length).")

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))


if FLAGS.experiment == 'all':
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    with open(os.getcwd() + '/data/clean/shuffle.pkl', 'wb') as fp:
        pickle.dump(shuffle_indices, fp)

    #4000 testing articles
    x_shuffled = x_shuffled[:-4000]
    y_shuffled = y_shuffled[:-4000]

elif FLAGS.experiment == 'Trump':
    idx_trump = [idx for idx, article in enumerate(x_origin) if ('trump' not in article) and ('Trump' not in article) and ('TRUMP' not in article)]
    x = x[idx_trump]
    y = y[idx_trump]

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]


# Split train/test set
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
# what is {:d}?
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    #launches graph in a session.
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # parameters for CNN for text classification, below
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=embedding_dimension,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # is cnn.learning_rate the standard cnn learning rate?
        optimizer = tf.train.AdamOptimizer(cnn.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.experiment))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        #from here to line 203 added
        sess.run(tf.global_variables_initializer())
        if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
            vocabulary = vocab_processor.vocabulary_
            initW = None
            if embedding_name == 'word2vec':
                # load embedding vectors from the word2vec
                print("Load word2vec file {}".format(cfg['word_embeddings']['word2vec']['path']))
                initW = data_helpers.load_embedding_vectors_word2vec(vocabulary,
                                                                     cfg['word_embeddings']['word2vec']['path'],
                                                                     cfg['word_embeddings']['word2vec']['binary'])
                print("word2vec file has been loaded")
            elif embedding_name == 'glove':
                # load embedding vectors from the glove
                print("Load glove file {}".format(cfg['word_embeddings']['glove']['path']))
                initW = data_helpers.load_embedding_vectors_glove(vocabulary,
                                                                  cfg['word_embeddings']['glove']['path'],
                                                                  embedding_dimension)
                print("glove file has been loaded\n")
            sess.run(cnn.W.assign(initW))

        # learning rate was added as one of the parameters
        def train_step(x_batch, y_batch, learning_rate):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
              cnn.learning_rate: learning_rate
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}, learning_rate {:g}".format(time_str, step, loss, accuracy, learning_rate))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            print("got here 0")
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            print("got here 1")
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            print("got here 2")
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # It uses dynamic learning rate with a high value at the beginning to speed up the training

        # So does this speed the max learning rate by 5%
        max_learning_rate = 0.005
        min_learning_rate = 0.0001
        decay_speed = FLAGS.decay_coefficient*len(y_train)/FLAGS.batch_size

        # Training loop. For each batch...
        counter = 0
        for batch in batches:
            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-counter/decay_speed)
            counter += 1
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch, learning_rate)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))