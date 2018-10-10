#! /usr/bin/env python

import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.nan)
import os
import sys
import csv
import interpret
import data_helpers
from sklearn import metrics
from tensorflow.contrib import learn
#import yaml
import pickle

maxpool_x = 2;
maxpool_y = 128;

def matrix_multiply(a, b):
    a=np.array(a)
    b=np.array(b)
    new_array = np.zeros((a.shape[0], b.shape[1]))
    for row in range(a.shape[0]):
        for col in range(b.shape[1]):
            weights_x_activation = np.multiply(a[row],b[:,col])
            element = sum(weights_x_activation)
            new_array[row][col]=element
    return new_array

def get_wi_ai(a,b):
    a=np.array(a)
    b=np.array(b)
    new_array = np.zeros((a.shape[0], b.shape[1]))
    batch_relevant=[]
    for row in range(a.shape[0]):
        relevant = np.zeros((maxpool_x, maxpool_y))
        for col in range(b.shape[1]):
            weights_x_activation = np.multiply(a[row],b[:,col])
            relevant[col]=weights_x_activation
        batch_relevant.append(relevant)
    return np.array(batch_relevant)

def softmax(x):
    """Compute softmax values for each sets of scores in x"""
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

cfg = {'word_embeddings': {'default': 'word2vec', 'word2vec':
    {'path': '/Users/sofia/Documents/src/fakenews1/GoogleNews-vectors-negative300.bin',
     'dimension': 300, 'binary': True}, 'glove': {'path': '../../data/glove.6B.100d.txt', 'dimension': 100, 'length': 400000}},
     'datasets': {'default': '20newsgroup', 'mrpolarity': {'positive_data_file': {'path': 'data/rt-polaritydata/rt-polarity.pos',
                                                                                  'info': 'Data source for the positive data'},
                                                           'negative_data_file': {'path': 'data/rt-polaritydata/rt-polarity.neg',
                                                                                  'info': 'Data source for the negative data'}},
                  '20newsgroup': {'categories': ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian'], 'shuffle': True, 'random_state': 42},
                  '20newsgroup': {'categories': ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian'], 'shuffle': True, 'random_state': 42},
                  'localdata': {'container_path': '../../data/input/SentenceCorpus', 'categories': None, 'shuffle': True, 'random_state': 42}}}

# Parameters
# ==================================================

# Data Parameters

# Eval Parameters
tf.flags.DEFINE_string("experiment", "all", "All subjects (all), Trump or Email.")
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_integer("y_test_special", 1,  "The y value for the specific x raw if there is one.")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

datasets = None

# Load data.
checkpoint_dir = os.getcwd() + '/runs/' + FLAGS.experiment + '/checkpoints/'
positive_data_file = os.getcwd() + '/data/clean/real.pkl'
negative_data_file = os.getcwd() + '/data/clean/fake.pkl'
dataset_name = cfg["datasets"]["default"]


x_origin, y_origin = data_helpers.load_data_and_labels(positive_data_file, negative_data_file)
#x_origin = x_origin[:1000]
#y_origin = y_origin[:1000]
y_test = np.argmax(y_origin, axis=1)
print("Total number of test examples: {}".format(len(y_test)))


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

x_raw = [" ".join(clean(x).split(" ")[:1000]) for x in x_origin]

# Map data into vocabulary
vocab_path = os.path.join(checkpoint_dir, "..", "vocab")
print(vocab_path)
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))


if FLAGS.experiment == 'all':
    with open(os.getcwd() + '/data/clean/shuffle.pkl', 'rb') as fp:
        shuffle_indices = pickle.load(fp)

    x_test = x_test[shuffle_indices]
    y_test = y_test[shuffle_indices]

    #4000 testing articles
    x_test = x_test[-4000:]
    y_test = y_test[-4000:]

    x_raw = np.array(x_raw)
    x_raw = x_raw[shuffle_indices]
    #4000 testing articles
    x_raw = x_raw[-4000:]

elif FLAGS.experiment == 'Trump':
    idx_trump = [idx for idx, article in enumerate(x_origin) if ((' trump ' in article) or (' Trump ' in article) or (' TRUMP ' in article))]
    x_test = x_test[idx_trump]
    y_test = y_test[idx_trump]
    x_raw = np.array(x_raw)
    x_raw = x_raw[idx_trump]

elif FLAGS.experiment == 'war':
    idx_email = [idx for idx, article in enumerate(x_origin) if (' war ' in article) or (' War ' in article) or (' WAR '  in article)]
    x_test = x_test[idx_email]
    y_test = y_test[idx_email]
    x_raw = np.array(x_raw)
    x_raw = x_raw[idx_email]

print("\nEvaluating...\n")
# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(os.getcwd() + '/runs/' + FLAGS.experiment + '/checkpoints/')
print("checkpoint_file", checkpoint_file)
graph = tf.Graph()
print("0")
with tf.device('/gpu:1'):
    with graph.as_default():
        print("1")
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        print("2")
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").o utputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            scores = graph.get_operation_by_name("output/scores").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            conv_mp3 = graph.get_operation_by_name("conv-maxpool-3/conv").outputs[0]
            relu_mp3 = graph.get_operation_by_name("conv-maxpool-3/relu").outputs[0]
            before_predictions=graph.get_operation_by_name("W").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
            b = graph.get_operation_by_name("output/b").outputs[0]
            pool_mp3 = graph.get_operation_by_name("conv-maxpool-3/pool").outputs[0]
            conv_lensequence = graph.get_operation_by_name("conv-maxpool-3/conv").outputs[0]
            h_drop = graph.get_operation_by_name("dropout/dropout/mul").outputs[0]
            embedding_W = graph.get_operation_by_name("embedding/W").outputs[0]

            # Collect the predictions here
            all_predictions = []
            all_probabilities = None

            all_x = []
            all_w = []

            all_wi_ai=np.zeros((0,maxpool_x,maxpool_y))

            best_trigrams ={}
            n=5
            all_top_n_neurons=[]
            ind=0
            for i,x_test_batch in enumerate(batches):
                batch_predictions_scores = sess.run([predictions, scores,conv_mp3,before_predictions,b,pool_mp3,h_drop,conv_lensequence,relu_mp3, embedding_W], {input_x: x_test_batch, dropout_keep_prob: 1.0})
                predictions_result = batch_predictions_scores[0]
                probabilities = softmax(batch_predictions_scores[1])
                weights = batch_predictions_scores[3]
                b_result=batch_predictions_scores[4]
                pool_post_relu = batch_predictions_scores[5]
                x_result = batch_predictions_scores[6]
                conv=batch_predictions_scores[7]
                relu_result = batch_predictions_scores[8]

                all_predictions = np.concatenate([all_predictions, batch_predictions_scores[0]])

                xW=np.matmul(x_result,weights)
                batch_wi_ai = get_wi_ai(x_result, weights)
                all_wi_ai = np.concatenate([all_wi_ai, batch_wi_ai])

                embedding_W_result = batch_predictions_scores[9]


                best_trigrams, top_n_neurons = interpret.interpret_many(x_raw[i * FLAGS.batch_size:i * FLAGS.batch_size + FLAGS.batch_size + 1], relu_result, pool_post_relu, batch_wi_ai, best_trigrams, n=n)
                all_top_n_neurons+=top_n_neurons
                if all_probabilities is not None:
                    all_probabilities = np.concatenate([all_probabilities, probabilities])
                else:
                    all_probabilities = probabilities
            
# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    #for each thing in all predictions, if its equal to y_test you wanna print x_raw

    wrong = [(x_raw[i],y_test[i]) for i in range(len(y_test)) if all_predictions[i]!=y_test[i]]

    with open(checkpoint_dir+"false_pos.txt", 'w') as false_pos, open(checkpoint_dir+'false_neg.txt', 'w') as false_neg:
        for headline, num in wrong:
            if num==0:
                # should be fake, but was real
                false_pos.write(headline+"\n")
            else:
                #should be real, but was fake
                false_neg.write(headline+"\n")

    correct = [(x_raw[i],y_test[i]) for i in range(len(y_test)) if all_predictions[i]==y_test[i]]
    with open(checkpoint_dir+"true_pos.txt", 'w') as true_pos, open(checkpoint_dir+'true_neg.txt', 'w') as true_neg:
        for headline, num in correct:
            if num==1:
                true_pos.write(headline+"\n")
            else:
                true_neg.write(headline+"\n")
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
    print(metrics.confusion_matrix(y_test, all_predictions))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw),
                                              [int(prediction) for prediction in all_predictions],
                                              [ "{}".format(probability) for probability in all_probabilities]))
out_path = os.path.join(checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)

def write_trigram_dict(filename, dictionary):
    with open(filename, 'w') as f: 
        for k in dictionary.keys():
            list_o_lists=dictionary[k]
            best_trigrams_for_k=[]
            for li in list_o_lists:
                if len(li[1])>0:
                    trigram = ' '.join(li[1][0])
                else: 
                    trigram = ' '.join(li[1])
                best_trigrams_for_k.append(trigram)
            f.write("i: "+str(k)+'\n')
            f.write("trigrams: ")
            for trigram in best_trigrams_for_k:
                f.write(trigram+",")
            f.write('\n')

def first_element_from_tuples(tuple_list):
    return [element[0] for element in tuple_list]

best_n_trigrams = interpret.get_best_n_for_each_neuron(best_trigrams, 15)

import pickle

with open(checkpoint_dir+"best_trigrams_pickle.txt", 'wb') as f2:
    pickle.dump(best_trigrams, f2)

write_trigram_dict(checkpoint_dir+'best_trigrams.txt',best_trigrams)
write_trigram_dict(checkpoint_dir+'best_n_trigrams.txt',best_n_trigrams)

best_neurons_fake, best_neurons_real, worst_neurons_fake, worst_neurons_real = interpret.get_n_best_neurons(weights, 30)
best_fake_neurons = {key : best_n_trigrams[key] for key in first_element_from_tuples(best_neurons_fake)}
best_real_neurons = {key: best_n_trigrams[key] for key in first_element_from_tuples(best_neurons_real)}
worst_fake_neurons = {key: best_n_trigrams[key] for key in first_element_from_tuples(worst_neurons_fake)}
worst_real_neurons = {key: best_n_trigrams[key] for key in first_element_from_tuples(worst_neurons_real)}
write_trigram_dict(checkpoint_dir+'best_n_fake_neurons.txt',best_fake_neurons)
write_trigram_dict(checkpoint_dir+'worst_n_fake_neurons.txt', worst_fake_neurons)
write_trigram_dict(checkpoint_dir+'best_n_real_neurons.txt', best_real_neurons)
write_trigram_dict(checkpoint_dir+'worst_n_real_neurons.txt',worst_real_neurons)

with open(checkpoint_dir+"all_top_n_neurons.txt", 'wb') as f:
    pickle.dump(all_top_n_neurons, f)

np.save(checkpoint_dir+"weights",weights)
np.save(checkpoint_dir+"all_wi_ai", all_wi_ai)
