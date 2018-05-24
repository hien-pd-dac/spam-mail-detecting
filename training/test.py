from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

y_true = [0, 1, 2, 0, 1, 1]
y_pred = [0, 1, 1, 0, 0, 1]

print(recall_score(y_true, y_pred, labels=[1], average="macro"))

