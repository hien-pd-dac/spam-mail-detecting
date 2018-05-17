from __future__ import print_function
import numpy as np
from sklearn import neighbors
from sklearn.metrics import accuracy_score  # for evaluating results

# data path and file name
path = "ex6DataPrepared/"

train_data_fn = 'train_features.txt'
test_data_fn = "test_features.txt"
train_label_fn = 'train_labels.txt'
test_label_fn = 'test_labels.txt'

N_WORDS = 3000
K_NEIGHBORS = 13


def read_data(data_fn, label_fn):
    # read label_fn
    with open(path + label_fn) as f:
        content = f.readlines()
    label = [int(x.strip()) for x in content]

    # read data_fn
    with open(path + data_fn) as f:
        content = f.readlines()
    table_data = []
    for line in content:
        line = line.strip()
        row_data = line.split(" ")
        table_data.append(row_data)

    return (table_data, label)


(train_data, train_label) = read_data(train_data_fn, train_label_fn)
(test_data, test_label) = read_data(test_data_fn, test_label_fn)

clf = neighbors.KNeighborsClassifier(n_neighbors=K_NEIGHBORS, weights='distance')
clf.fit(train_data, train_label)

y_pred = clf.predict(test_data)
print("Training size = %d, accuracy = %.2f%%" %
      (len(train_label), accuracy_score(test_label, y_pred)*100))

