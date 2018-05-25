import numpy as np
from os import listdir
from os.path import join, dirname
from collections import Counter
import os


def extract_feature(dataset):
    # try catch
    dictionary_file = open('dictionary.txt', 'r')
    dictionary = dictionary_file.read().split()
    dictionary_file.close()

    feature_file = open(dataset + '_features.txt', 'w')
    label_file = open(dataset + '_labels.txt', 'w')
    data_path = os.getcwd() + '/../preprocessing-data/processed-data/'

    file_names = listdir(data_path)
    stt = 0
    table_data = []
    for file_name in file_names:
        row_data = []
        stt += 1
        email_file = open(join(data_path, file_name), 'r')
        words = email_file.read().split()
        if 'spam' in file_name:
            label_file.write('1\n')
        else:
            label_file.write('0\n')

        for element in dictionary:
            num = words.count(element)
            feature_file.write("" + str(num) + " ")

        feature_file.write("\n")
        email_file.close()

    feature_file.close()
    label_file.close()


extract_feature('data')

