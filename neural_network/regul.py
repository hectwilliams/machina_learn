#!/usr/bin/env python 

"""
    Use Dropout (MCDropout)

    Usage: regul.py
"""

import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt 
import keras 
from functools import partial 
import sys 
import numpy as np

current_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)) ) 
fashion_mnist = keras.datasets.fashion_mnist
epochs = 10

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# create cross validation set 
validation_set_size = 1000
x_valid, x_train = x_train[:validation_set_size], x_train[validation_set_size:]
y_valid, y_train = y_train[:validation_set_size], y_train[validation_set_size:]

# normalize (scale) pixel intensities 
x_valid = x_valid / 255.0
x_train = x_train / 255.0
x_test  = x_test / 255.0

# reshape array of (1,28,28) to (28, 28)

x_valid = np.reshape(x_valid, (-1, 28, 28))
x_train = np.reshape(x_train, (-1, 28, 28))
x_test = np.reshape(x_test, (-1, 28, 28))

# sys.exit()

class_labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Dropout example 

RegularizedDense = partial(keras.layers.Dense, activation="elu", kernel_initializer="he_normal", use_bias=False )

layers = [
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dropout(rate=0.2),
    RegularizedDense(300),
    keras.layers.Dropout(rate=0.2),
    RegularizedDense(100),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(10, activation="softmax")
]

# dropping off when testing test set
model = keras.models.Sequential(layers)
model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=0.1, decay=1/5), metrics=["accuracy"] )
model.fit(x_train, y_train, epochs=epochs, validation_data=(x_valid, y_valid))

#
y_predict_no_drop = np.round(model.predict([x_test[:1]]) , 2)# first instance in test set
y_test[0] # 9
print(np.round(y_predict_no_drop,4)) # [[1.5083739e-06 9.8573755e-06 1.9047060e-06 3.4041141e-06 2.6524012e-065.4067850e-02 7.6042052e-06 1.2994860e-01 8.7112701e-04 8.1508547e-01]]

# dropping enabled when testing test set 
y_probas = np.stack( [ model(x_test, training=True) for _ in range(100)] ) # 100 predictions on test set 
print("First ten predictions of instance 1")
print(np.round(y_probas[0][:10], 2))
y_prob_avg = y_probas.mean(axis=0) #
print("Average prediction of instance 1 ( mean of distribution ) ")
print(np.round(y_prob_avg[:1], 2)) 
print(" std deviation of predictions of instance 1")
y_prob_std = np.round(y_probas.std(axis = 0), 2)
print(y_prob_std[:1])