from __future__ import print_function
import time
import os
from sklearn import neighbors
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score  # for evaluating results
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

data_feature_fn = "../knn_feature-extraction/data_features.txt"

data_label_fn = "../knn_feature-extraction/data_labels.txt"

path = os.getcwd() + "/"

N_WORDS = 3000
K_NEIGHBORS = 10


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

    return table_data, label


(X_data, y_label) = read_data(data_feature_fn, data_label_fn)

X_train, X_test, y_train, y_test = train_test_split(
     X_data, y_label, test_size=0.3, stratify=y_label, random_state=42)

# X_train, X_test, y_train, y_test = train_test_split(
#      train_data, train_label, test_size=0.5, stratify=train_label)

clf = neighbors.KNeighborsClassifier(n_neighbors=K_NEIGHBORS, weights='distance')
clf.fit(X_train, y_train)

start_time = time.time()

y_pred = clf.predict(X_test)

pred_time = time.time() - start_time

print("Number of neighbors: k = %d" % K_NEIGHBORS)

print("Training size = %d" % len(y_train))

print("Test size = %d" % len(y_test))

print("Time for predict = %ds" % pred_time)

print("Accuracy = %.2f%%" %
      (accuracy_score(y_test, y_pred)*100))

print("Precision of spam = %.2f%%" %
      (precision_score(y_test, y_pred, labels=[1], average="macro")*100))

print("Precision of non-spam = %.2f%%" %
      (precision_score(y_test, y_pred, labels=[0], average="macro")*100))

print("Recall of spam = %.2f%%" %
      (recall_score(y_test, y_pred, labels=[1], average="macro")*100))

print("Recall of non-spam = %.2f%%" %
      (recall_score(y_test, y_pred, labels=[0], average="macro")*100))
