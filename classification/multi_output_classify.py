import classify_helper 
import numpy as np 
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier

rng = np.random.default_rng(seed=42)

# mnist dataset 

X_train, X_test, y_train, y_test = classify_helper.get_mnist()

# train
m_row = len(X_train) # number of instance provided from helper, not the object property size created by dataset
n_row = X_train.shape[1]
noise = rng.integers(low=0, high=100, size=  (m_row,n_row ) )
x_train_noise = X_train + noise
y_train_orginal = X_train

# test 
m_row = len(X_test)
n_row = X_test.shape[1]
noise =  rng.integers(low=0, high=100, size= (m_row,n_row ) )
x_test_noise = X_test + noise
y_test_orginal = X_test

# estimator 
knn_clf = KNeighborsClassifier()

# train
knn_clf.fit(x_train_noise, y_train_orginal)

# predict (noisy or transformed image could originally turned into)
some_digit = x_train_noise.iloc[[0]]
some_digit_predict = knn_clf.predict(some_digit)

classify_helper.plot_image(some_digit.to_numpy())
classify_helper.plot_image(some_digit_predict)

