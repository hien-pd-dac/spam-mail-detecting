import re
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from os import listdir
from os.path import isfile, join

# file = open('raw-data/part1/3-1msg1.txt', 'r')

def remove_stopwords(file_content):
	words = re.sub(r'[.!,;?"#$%&\'()*+-/@<>:0-9{}\[\]/=~_`|\\]', ' ', file_content).split()
	# words = list(set(words) - set(stopwords.words('english')))
	words_without_stopwords = [word.lower() for word in words if word not in stopwords.words('english')]
	return words_without_stopwords

def lemmatize(words):
	lmtzr = WordNetLemmatizer()
	# lemmatized_words = [lmtzr.lemmatize(word.lower()) for word in words]
	lemmatized_words = [lmtzr.lemmatize(word, pos[0].lower()) if  pos[0].lower() in ['a','n','v'] else lmtzr.lemmatize(word) for word, pos in pos_tag(words)]
	return lemmatized_words
		

# print(lemmatize(remove_stopwords(file.read())))
# print(stopwords.words('english'))
def process(read_path, write_path):
	ham_count = 0
	spam_count = 0
	file_names = listdir(read_path)
	for file_name in file_names:
		try:
			read_file = open(read_path + '/' + file_name, 'r')
		except IOError:
			print("Error: File " + "file_name" + " does not appear to exist.")
		print("Processing "+ read_path + "/" + file_name + " ...")

		if 'spm' in file_name:
			write_file = open(write_path + '/spam' + str(spam_count) + ".txt", 'w')
			spam_count += 1
		else:
			write_file = open(write_path + '/ham' + str(ham_count) + '.txt', 'w')
			ham_count += 1

		write_file.write(' '.join(lemmatize(remove_stopwords(read_file.read()))))

		read_file.close()
		write_file.close()

process('raw-data/train', 'pocessed-data/train')
process('raw-data/test', 'pocessed-data/test')