import re
import csv

import collections
import codecs
import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import tensorflow as tf
import sys
import os

tf.flags.DEFINE_string("experiment", "all", "All subjects (all), Trump.")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

directory = os.getcwd() + '/runs/' + FLAGS.experiment + '/checkpoints/'




d = collections.OrderedDict()
vectorizer = CountVectorizer()

NUM = 1000

def clean(text):
    text = re.sub(r'\([^)]*\)', '', text)
    text = ' '.join([s for s in text.split() if not any([c.isdigit() for c in s])])
    text = ' '.join([s for s in text.split() if not any([not c.isalpha() for c in s])]) 
    return text

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

real_titles = open(directory+'most_real.txt').readlines()+open(directory+'least_fake.txt').readlines()
fake_titles = open(directory+'most_fake.txt').readlines()+open(directory+'least_real.txt').readlines()

real_titles = [clean_str(clean(title))[:-1] for title in real_titles]
fake_titles = [clean_str(clean(title))[:-1] for title in fake_titles]
x = real_titles+fake_titles
y=['real' for real in real_titles]+['fake' for fake in fake_titles]

x=np.array(x)
y=np.array(y)
dev_sample_percentage=1
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

X=vectorizer.fit_transform(x)

matrix = vectorizer.transform(real_titles)
freqs = [(word, matrix.getcol(idx).sum()) for word, idx in vectorizer.vocabulary_.items()]
#sort from largest to smallest
sorted_words_true=(sorted (freqs, key = lambda x: -x[1])[:NUM])
print (sorted_words_true)

matrix = vectorizer.transform(fake_titles)
freqs = [(word, matrix.getcol(idx).sum()) for word, idx in vectorizer.vocabulary_.items()]
#sort from largest to smallest
sorted_words_false=(sorted (freqs, key = lambda x: -x[1])[:NUM])
print (sorted_words_false)

unique_true=np.setdiff1d([word for (word, number) in sorted_words_true],[word for (word, number) in sorted_words_false])
unique_false=np.setdiff1d([word for (word, number) in sorted_words_false],[word for (word, number) in sorted_words_true])

print("unique true \n",unique_true)
print()
print("unique false \n", unique_false)


import nltk

real = unique_true

wordtags = nltk.ConditionalFreqDist((w.lower(), t) for w, t in nltk.corpus.brown.tagged_words(tagset="universal"))
real_dict={}
for word in real:
	#print(word+"\n")
	if word in wordtags:
		if wordtags[word].max() in real_dict:
			real_dict[wordtags[word].max()]+=[word]
		else:
			real_dict[wordtags[word].max()]=[word]
	else:
		if 'N/A' in real_dict:
			real_dict['N/A']+=[word]
		else:
			real_dict['N/A']=[word]


fake = unique_false
fake_dict={}
for word in fake:

        if word in wordtags:
                if wordtags[word].max() in fake_dict:
                        fake_dict[wordtags[word].max()]+=[word]
                else:
                        fake_dict[wordtags[word].max()]=[word]
        else:
                if 'N/A' in fake_dict:
                        fake_dict['N/A']+=[word]
                else:
                        fake_dict['N/A']=[word]

#print(fake_dict)

only_real=[]
for key in real_dict:
	if key in fake_dict:
		print("KEY: ",key)
		print(real_dict[key])
		print(fake_dict[key])
	else:
	#	print("only real")
		only_real.append((key,real_dict[key]))
for thing in only_real:
	print("only real")
	print (thing)

for key in fake_dict:
	if key not in real_dict:
		print("only fake")
		print(key)
		print (fake_dict[key])

import nltk
real = unique_true
wordtags = nltk.ConditionalFreqDist((w.lower(), t) for w, t in nltk.corpus.brown.tagged_words(tagset="universal"))
real_dict={}