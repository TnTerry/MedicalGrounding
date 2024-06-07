import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
X, y = make_multilabel_classification(random_state=0)
# clf = MultiOutputClassifier(clf).fit(X, y)
# # get a list of n_output containing probability arrays of shape
# # (n_samples, n_classes)
# y_pred = clf.predict_proba(X)
# # extract the positive columns for each output
# y_pred = np.transpose([pred[:, 1] for pred in y_pred])
# roc_auc_score(y, y_pred, average=None)
# from sklearn.linear_model import RidgeClassifierCV
# clf = RidgeClassifierCV().fit(X, y)
# roc_auc_score(y, clf.decision_function(X), average=None)
print(y)