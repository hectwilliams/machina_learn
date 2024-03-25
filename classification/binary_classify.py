import sys 
import numpy as np 
import pandas as pd 
import classification.classify_helper as classify_helper
import matplotlib as mpl
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# mnist dataset 

X_train, X_test, y_train, y_test = classify_helper.get_mnist()

# 5 digit classifier 

y_train_5_classifier = (y_train == 5)
y_test_5_classifier = (y_test == 5)


# y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5_classifier, cv=3) # classifies each instance during cross-validation

# conf_matrix = confusion_matrix(y_train_5_classifier, y_train_pred)
# row - actual class 
# col - predicted class 
#
#  [                    non 5 images        5 inages        
#   non 5 images            [53892           687]
#   5 images                [1891           3530]
#  ]


sgd_clf = SGDClassifier(random_state=42)

y_decision_scores = cross_val_predict(sgd_clf, X_train, y_train_5_classifier, cv=3, method='decision_function') # each instance is returned a decision score swayed by threshold
precisions, recalls, thresholds = precision_recall_curve(y_train_5_classifier, y_decision_scores)

# find index of first occurence of precision >= 90 percent 
precision_90_percent_idx = np.argmax(precisions >= 0.90)
# get precision
precision_90_percent_precision = precisions[precision_90_percent_idx]
# get threshold at index 
threhold_90_percent_precision = thresholds[precision_90_percent_idx]
# get recall
recall_90_percent_precision = recalls[precision_90_percent_idx]
# classify ( i.e. 5-image or non-5-image ) or predict using threshold
y_train_prediction_90_percent_precision = (y_decision_scores >= threhold_90_percent_precision)

precision_score_ = precision_score(y_train_5_classifier, y_train_prediction_90_percent_precision)
recall_score_ = recall_score(y_train_5_classifier, y_train_prediction_90_percent_precision)
roc_auc_score_ = roc_auc_score(y_train_5_classifier, y_decision_scores)
fpr_sgd, tpf_sgd, thresholds_sgd = roc_curve(y_train_5_classifier, y_decision_scores) # Receiver Operating Characteristic (ROC)

classify_helper.precision_vs_recall_vs_threshold(precisions, recalls, thresholds, [ [threhold_90_percent_precision, recall_90_percent_precision, precision_90_percent_precision] ])
print("SGD\n\nprecision score = \t {}\nrecall score = \t {}\t roc_auc_score (threshold = 0) = {}".format(precision_score_, recall_score_,roc_auc_score_ ))


forest_clf = RandomForestClassifier(random_state=42)

y_predict_probability = cross_val_predict(forest_clf, X_train, y_train_5_classifier, cv=3, method="predict_proba")
y_positive_class_scores = y_predict_probability[:, 1] # classification scores of potential 5-image
fpr_forest, tpr_forest, threshold_forest = roc_curve(y_train_5_classifier, y_positive_class_scores) # Receiver Operating Characteristic (ROC)
roc_auc_score_ = roc_auc_score(y_train_5_classifier, y_positive_class_scores)
# recall_score_ = recall_score(y_train_5_classifier, y_positive_class_scores)
# precision_score_ = precision_score(y_train_5_classifier, y_positive_class_scores)

classify_helper.plot_roc_curve(fpr=[fpr_sgd, fpr_forest], tpr=[tpf_sgd, tpr_forest], labels=["SGD", "RANDOM FOREST"], trace_color=["r--", "b--"] )
print("RANDOM FORST\n\nprecision score = \t {}\nrecall score = \t {}\t roc_auc_score={}".format(precision_score_, recall_score_, roc_auc_score_ ))

plt.show()