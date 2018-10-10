import numpy as np
#import matplotlib.pyplot as plt

def interpret_many(x_raw, relu, pool, all_wi_ai, best_trigrams = {}, n=5):
	pool = pool.squeeze() #should be len(x_raw) x num_filters (128)
	relu = relu.squeeze()
	if len(x_raw)==1:
		pool=np.expand_dims(pool, axis=0)
		relu=np.expand_dims(relu, axis=0)
	text_lists = []
	for text in x_raw:
		text_list = text.split()
		text_lists.append(text_list)
	top_n_neurons = []
	for i in range(pool.shape[0]):
		best_trigrams = interpret(text_lists[i],relu[i], pool[i], best_trigrams)
		weights = all_wi_ai[i].T #2 x 128 --> 128 x 2
		top_n_neurons.append([get_n_best_neurons(weights,5)])
	return best_trigrams, top_n_neurons


def find_relu_index(relu, pooled_val, i):
	#each thing in relu should be 128 length
	#index represents the index (out of max seq len) that resulted in the pooled val
	for ind, arr in enumerate(relu):
		if arr[i]==pooled_val:
			return ind
	return None

def interpret(word_list, relu, pool, best_trigrams={}):
	relu = relu.squeeze()
	for i, pooled_val in enumerate(pool):
		relu_index = find_relu_index(relu, pooled_val, i)
		trigram = word_list[relu_index:relu_index+3]
		if i in best_trigrams:
			best_trigrams[i]+=[(pooled_val,[trigram])]
		else:
			best_trigrams[i]=[(pooled_val,[trigram])]
	return best_trigrams

def get_best_n_for_each_neuron(best_trigrams,n):
	best_n_trigrams=best_trigrams.copy()
	for neuron in best_trigrams.keys():
		best_n_trigrams[neuron]=sorted(best_trigrams[neuron])[:n]
	return best_n_trigrams

def make_weight_histogram(weights):
	plt.figure(1)
	plt.subplot(211)
	plt.title("Weights for Fake News Indicator")
	plt.hist(weights[:,0], bins = 20, range = [-0.3,0.3])

	plt.subplot(212)
	plt.title("Weights for Real News Indicator")
	plt.hist(weights[:,1], bins=20, range = [-0.3,0.3])
	plt.show()

def get_most_relevant_neurons(all_wi_ai=None, ind = None, abs = False):
	pass
	if ind is None:
		fake_news = np.mean(all_wi_ai[:,0,:], axis = 0)
		real_news = np.mean(all_wi_ai[:,1,:], axis = 1)


def make_wi_ai_histogram(all_wi_ai, ind = None):
	if ind is None:
		#plot the average
		fake_news = np.mean(all_wi_ai[:,0,:], axis = 0)
		real_news = np.mean(all_wi_ai[:,1,:], axis = 1)
	else:
		wi_ai = all_wi_ai[ind]
		#plot x_raw[ind] weights*activation for fake news indicator
		fake_news=wi_ai[0]
		#plot x_raw[ind] weights*activation for real news indicator
		real_news=wi_ai[1]
	plt.figure(1)
	plt.subplot(211)
	plt.title("W_i * a_i for Fake News Indicator")
	plt.hist(fake_news)

	plt.subplot(212)
	plt.title("W_i * a_i for Real News Indicator")
	plt.hist(real_news)
	plt.show()

def make_list_of_arrays_into_list(l):
	total_arr = []
	for array in l:
		for num in array:
			total_arr.append(num)
	return total_arr

def label_peaks(ax, li):
	for i in range(129):
		count = li.count(i)
		if count>5:
			ax.annotate(i, xy=(i,count))

def make_top_neuron_histogram(all_top_neurons):
	fake_news_pos = make_list_of_arrays_into_list([top_n[0][0] for top_n in all_top_neurons])
	real_news_pos = make_list_of_arrays_into_list([top_n[0][1] for top_n in all_top_neurons])
	fake_news_neg = make_list_of_arrays_into_list([top_n[0][2] for top_n in all_top_neurons])
	real_news_neg = make_list_of_arrays_into_list([top_n[0][3] for top_n in all_top_neurons])

	# from data helpers: plt.figure(1)
	ax = plt.subplot(221)
	plt.title("Most positive for Fake News Indicator")
	plt.xticks(np.arange(0,129,1))
	plt.ylabel('count')
	plt.xlabel('neuron number')
	plt.hist(fake_news_pos, bins=128)
	labels = [str(i) if i%10==0 else '' for i in range(129)]
	ax.set_xticklabels(labels)
	label_peaks(ax,fake_news_pos)

	ax2 = plt.subplot(222)
	plt.title("Most positive for Real News Indicator")
	plt.ylabel('count')
	plt.xlabel('neuron number')
	plt.xticks(np.arange(0,129,1))
	plt.hist(real_news_pos, bins = 128)
	labels = [str(i) if i%10==0 else '' for i in range(129)]
	ax2.set_xticklabels(labels)
	label_peaks(ax2,real_news_pos)

	ax3 = plt.subplot(223)
	plt.title("Most negative for Fake News Indicator")
	plt.ylabel('count')
	plt.xlabel('neuron number')
	plt.xticks(np.arange(0,129,1))
	plt.hist(fake_news_neg, bins = 128)
	labels = [str(i) if i%10==0 else '' for i in range(129)]
	ax3.set_xticklabels(labels)
	label_peaks(ax3,fake_news_neg)

	ax4 = plt.subplot(224)
	plt.title("Most negative for Real News Indicator")
	#plt.hist(real_news_neg, bins = 128)
	plt.ylabel('count')
	plt.xlabel('neuron number')
	plt.xticks(np.arange(0,129,1))
	plt.hist(real_news_neg, bins = 128)
	labels = [str(i) if i%10==0 else '' for i in range(129)]
	ax4.set_xticklabels(labels)
	label_peaks(ax4,real_news_pos)
	plt.savefig('Most_relevant_neurons.png')
	plt.show()
    # positive_labels = [[0, 1] for _ in positive_examples]
    # negative_labels = [[1, 0] for _ in negative_examples]

def get_n_best_neurons(weights, n,abs_value = False):
	#print(weights, weights.shape) #128 x 2
	arr_0 = weights[:,0]
	list_0=arr_0.argsort()[-n:][::-1]
	list_0 = [(element, round(arr_0[element],2)) for element in list_0]
	list_0_neg = arr_0.argsort()[:n]
	list_0_neg = [(element, round(arr_0[element],2)) for element in list_0_neg]
	arr_1 = weights[:,1]
	list_1=arr_1.argsort()[-n:][::-1]
	list_1 = [(element, round(arr_1[element],2)) for element in list_1]
	list_1_neg = arr_1.argsort()[:n]
	list_1_neg = [(element, round(arr_1[element],2)) for element in list_1_neg]
	#return weights for fake news, weights for real news
	return list_0, list_1, list_0_neg, list_1_neg

def get_info(ind, all_wi_ai, all_top_neurons, best_trigrams, cur_dir=''):
	import pickle
	print(all_top_neurons[ind], "is all top neurons[ind]")
	all_triples = []
	for li in all_top_neurons[ind][0]:
		triple_li = []
		for tup in li:
			neuron = tup[0]
			trigram = ' '.join(best_trigrams[neuron][ind][1][0])
			triple_li.append((neuron, trigram, tup[1]))
		all_triples.append(triple_li)
	wi_ais = all_wi_ai[ind]
		#plot x_raw[ind] weights*activation for fake news indicator
	fake_news=wi_ais[0]
		#plot x_raw[ind] weights*activation for real news indicator
	real_news=wi_ais[1]

	most_fake_indices=fake_news.argsort()[-10:][::-1]
	least_fake_indices=fake_news.argsort()[:10]
	most_real_indices = real_news.argsort()[-10:][::-1]
	least_real_indices = real_news.argsort()[:10]

	most_fake_trigrams=[]
	for neuron in most_fake_indices:
		trigram = ' '.join(best_trigrams[neuron][ind][1][0])
		string = trigram+', *neuron: '+str(neuron)+' '+str(fake_news[neuron])
		most_fake_trigrams.append(string)
	print("MOST FAKE: ",most_fake_trigrams)

	most_real_trigrams=[]
	for neuron in most_real_indices:
		trigram = ' '.join(best_trigrams[neuron][ind][1][0])
		most_real_trigrams.append(trigram+', *neuron: '+str(neuron)+' '+str(real_news[neuron]))
	print("MOST REAL: ",most_real_trigrams)

	least_fake_trigrams=[]
	for neuron in least_fake_indices:
		trigram = ' '.join(best_trigrams[neuron][ind][1][0])
		least_fake_trigrams.append(trigram+', *neuron: '+str(neuron)+' '+str(fake_news[neuron]))
	print("LEAST FAKE: ",least_fake_trigrams)

	least_real_trigrams=[]
	for neuron in least_real_indices:
		trigram = ' '.join(best_trigrams[neuron][ind][1][0])
		least_real_trigrams.append(trigram+', *neuron: '+str(neuron)+' '+str(real_news[neuron]))
	print("LEAST REAL: ",least_real_trigrams)


def get_info2(ind, all_wi_ai, all_top_neurons, best_trigrams):
    wi_ais = all_wi_ai[ind]
    # plot x_raw[ind] weights*activation for fake news indicator
    fake_news = wi_ais[0]
    # plot x_raw[ind] weights*activation for real news indicator
    real_news = wi_ais[1]

    most_fake_indices = fake_news.argsort()[-10:][::-1]
    least_fake_indices = fake_news.argsort()[:10]
    most_real_indices = real_news.argsort()[-10:][::-1]
    least_real_indices = real_news.argsort()[:10]
    most_fake_trigrams = []
    for neuron in most_fake_indices:
        trigram = ' '.join(best_trigrams[neuron][ind][1][0])
        # string = trigram+', *neuron: '+str(neuron)+' '+str(fake_news[neuron])
        most_fake_trigrams.append(trigram)

    # print(most_fake_trigrams)

    most_real_trigrams = []
    for neuron in most_real_indices:
        trigram = ' '.join(best_trigrams[neuron][ind][1][0])
        # most_real_trigrams.append(trigram+', *neuron: '+str(neuron)+' '+str(real_news[neuron]))
        most_real_trigrams.append(trigram)

    # print(most_real_trigrams)

    least_fake_trigrams = []
    for neuron in least_fake_indices:
        trigram = ' '.join(best_trigrams[neuron][ind][1][0])
        # least_fake_trigrams.append(trigram+', *neuron: '+str(neuron)+' '+str(fake_news[neuron]))
        least_fake_trigrams.append(trigram)

    # print(least_fake_trigrams)

    least_real_trigrams = []
    for neuron in least_real_indices:
        trigram = ' '.join(best_trigrams[neuron][ind][1][0])
        # least_real_trigrams.append(trigram+', *neuron: '+str(neuron)+' '+str(real_news[neuron]))
        least_real_trigrams.append(trigram)
    # print(least_real_trigrams)
    return most_fake_trigrams, most_real_trigrams, least_fake_trigrams, least_real_trigrams

def make_top_5_histogram():
	import pickle
	cur_dir = ""
	all_top_neurons="all_top_n_neurons.txt"
	with open(cur_dir+all_top_neurons, 'rb') as f:
		all_top_neurons = pickle.load(f)
	make_top_neuron_histogram(all_top_neurons)
