import sys 
import numpy as np 
import pandas as pd 
from sklearn.svm import SVC 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from matplotlib import pyplot as plt, colormaps
import  classify_helper 

# mnist dataset 
X_train, X_test, y_train, y_test = classify_helper.get_mnist()

## estimator ( support vector machine )
svm_clf = SVC() 

## train
# svm_clf.fit(X_train, y_train)

## quick test an instance 
# some_digit = X_train.iloc[[0]]
# predict_some_digit = svm_clf.predict(some_digit)
# scores_some_digit = svm_clf.decision_function(some_digit)

## train - predict_on_all_instances using cross-val
y_train_predict = cross_val_predict(svm_clf, X_train, y_train, cv=3)

## scores (per class)
prec_score = precision_score(y_train, y_train_predict, average=None)
rec_score = recall_score(y_train, y_train_predict, average=None)
print("precision \t{}\t\t\n recall\t{}\t\t".format(prec_score, rec_score))

## confusion matrix 
conf_matrix = confusion_matrix(y_train, y_train_predict)
plt.matshow(conf_matrix, cmap= colormaps["gray"])

## normalize matrix
row_sums = conf_matrix.sum(axis=1, keepdims=True)
norm_conf_mx = conf_matrix / row_sums
np.fill_diagonal(norm_conf_mx, 0) # fill diagonal with zeros 

## plot normalized confusion matrix 
plt.matshow(norm_conf_mx, cmap= colormaps["gray"])
plt.show()

## evaluate classifier (score per cross_validation fold)
# eval_cross_val_score = cross_val_score(svm_clf, X_train, y_train, cv=3, scoring="accuracy")
# print(eval_cross_val_score)

## standardize
# X_train_scaled = StandardScaler().fit_transform(X_train.astype(np.float64))
## evaluate classifier 
# eval_cross_val_score = cross_val_score(svm_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
# print(eval_cross_val_score)



