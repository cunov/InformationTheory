import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix 
from scipy.stats import rv_discrete
from datetime import datetime
import sys
import re

corpus_choices = ['before','during','after']
k_range = range(1,11)

def print_occurrences(test,corpus,length=40):
	x=[corpus[m.start():m.start()+length] for m in re.finditer(test,corpus)]
	if len(x) == 0:
		print('No matches')
	else:
		for i in x:
			print(i)

def weighted_choice(distinct_words,likelihoods):
	distinct_words_idx = list(range(len(distinct_words)))
	likelihoods = tuple(likelihoods[0])
	random_variable = rv_discrete(values=(distinct_words_idx,likelihoods))
	next_word_idx = int(random_variable.rvs(size=1))
	
	return distinct_words[next_word_idx]
	
def sample_next_word_after_sequence(word_sequence, alpha):
	try:
		tmp = k_words_idx_dict[word_sequence]
	except:
		print('No matches found for "{}"'.format(word_sequence))
		sys.exit(0)
	next_word_vector = next_after_k_words_matrix[k_words_idx_dict[word_sequence]] + alpha
	likelihoods = next_word_vector/next_word_vector.sum()
	
	return weighted_choice(distinct_words, likelihoods.toarray())
	
def stochastic_chain(seed, chain_length, alpha=0):
	seed_length = k
	current_words = seed.split(' ')
	if len(current_words) != seed_length:
		raise ValueError(f'wrong number of words, expected {seed_length}')
	sentence = seed

	for _ in range(chain_length):
		sentence+=' '
		next_word = sample_next_word_after_sequence(' '.join(current_words),alpha)
		sentence+=next_word
		current_words = current_words[1:]+[next_word]
	return sentence

def block_entropy(sets_of_k_words,next_after_k_words_matrix):
	col_sums = next_after_k_words_matrix.sum(axis=1)
	pxs = col_sums / len(sets_of_k_words)
	pxs = pxs.flatten().tolist()
	pxs = [item for sublist in pxs for item in sublist if item !=0]
	entropies = [-x*np.log2(x) for x in pxs]
	return sum(entropies)

df = pd.DataFrame(columns=['k','before','during','after'])
for k in k_range:
	entropies = {}
	for corpus_choice in corpus_choices:
		time_now = datetime.now()
		with open('Data/corpora/{}_corpus.txt'.format(corpus_choice),'r',encoding='utf-8') as f:
			corpus = f.read()
		for spaced in ['.','-',',','!','?','(','—',')','"']:
			corpus = corpus.replace(spaced, ' {0} '.format(spaced))

		corpus_words = corpus.split(' ')
		corpus_words = [word for word in corpus_words if word != '']
		distinct_words = list(set(corpus_words))
		print(len(distinct_words))
		word_idx_dict = {word:i for i, word in enumerate(distinct_words)}
		distinct_words_count = len(list(set(corpus_words)))

		
		sets_of_k_words = [ ' '.join(corpus_words[i:i+k]) for i, _ in enumerate(corpus_words[:-k]) ]
		sets_count = len(list(set(sets_of_k_words)))
		next_after_k_words_matrix = dok_matrix((sets_count,len(distinct_words)))
		distinct_sets_of_k_words = list(set(sets_of_k_words))
		k_words_idx_dict = {word: i for i, word in enumerate(distinct_sets_of_k_words)}

		for i, word in enumerate(sets_of_k_words[:-k]):
			word_sequence_idx = k_words_idx_dict[word]
			next_word_idx = word_idx_dict[corpus_words[i+k]]
			next_after_k_words_matrix[word_sequence_idx, next_word_idx] +=1
		entropy = block_entropy(sets_of_k_words,next_after_k_words_matrix)
		entropies[corpus_choice] = entropy
		elapsed_time = (datetime.now() - time_now).seconds
		print('({},{}) done after {}s'.format(corpus_choice,k,elapsed_time))
	df = df.append({'k':k,'before':entropies['before'],'during':entropies['during'],'after':entropies['after']},ignore_index=True)
	print(df)

df.to_csv('Data/dataframes/block_entropy.csv')