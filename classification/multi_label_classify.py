import classify_helper
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 

# dataset 
X_train, X_test, y_train, y_test = classify_helper.get_mnist()

# build labels
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

# model
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

# predict some on some_image_digit
some_img_digit = X_train.iloc[[0]]
some_img_digit_prediction = knn_clf.predict(some_img_digit)

print(y_train.iloc[[0]])
print( some_img_digit_prediction)
