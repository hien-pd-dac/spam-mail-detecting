import numpy as np
from os import listdir
from os.path import join, dirname
from collections import Counter
import os

def make_dictionary():
	all_words = []
	data_path = join(dirname(dirname(os.path.realpath(__file__))), 'preprocessing-data/pocessed-data/train')
	file_names = listdir(data_path)
	for file_name in file_names:
		file = open(join(data_path, file_name), 'r')
		words = file.read().split()
		all_words += words
		file.close()
	# dictionary = np.unique(dictionary)
	dictionary_freq = Counter(all_words).most_common(3000)
	dictionary = [key for key, value in dictionary_freq]
	return dictionary

dict_file = open('dictionary.txt', 'w')
dictionary = make_dictionary()
dict_file.write(' '.join(dictionary))
dict_file.close()
print(len(dictionary))