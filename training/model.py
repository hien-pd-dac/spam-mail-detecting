from __future__ import print_function
import numpy as np
from scipy.sparse import coo_matrix  # for sparse matrix
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score  # for evaluating results

# data path and file name
path = "ex6DataPrepared/"

train_data_fn = 'train_features.txt'
test_data_fn = "test_features.txt"
train_label_fn = 'train_labels.txt'
test_label_fn = 'test_labels.txt'

nwords = 3000


def read_data(data_fn, label_fn):
    # read label_fn
    with open(path + label_fn) as f:
        content = f.readlines()
    label = [int(x.strip()) for x in content]

    # read data_fn
    with open(path + data_fn) as f:
        content = f.readlines()
    # remove '\n' at the end of each line
    content = [x.strip() for x in content] 

    # create matrix number_of_line x 3, type int
    dat = np.zeros((len(content), 3), dtype=int)

    # enumerate(content) tra ve cac cap (i, line) voi: i - so lan xuat hien, line - element
    for i, line in enumerate(content): 
        a = line.split(' ')
        dat[i, :] = np.array([int(a[0]), int(a[1]), int(a[2])])
    
    # remember to -1 at coordinate since we're in Python
    # check this: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
    # for more information about coo_matrix function 
    data = coo_matrix((dat[:, 2], (dat[:, 0] - 1, dat[:, 1] - 1)),\
             shape=(len(label), nwords))
    return (data, label)

(train_data, train_label) = read_data(train_data_fn, train_label_fn)
(test_data, test_label) = read_data(test_data_fn, test_label_fn)


clf = MultinomialNB(alpha=1.0)
clf.fit(train_data, train_label)

y_pred = clf.predict(test_data)
print("Training size = %d, accuracy = %.2f%%" %
      (train_data.shape[0], accuracy_score(test_label, y_pred)*100))

