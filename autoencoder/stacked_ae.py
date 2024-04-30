""""Creates a stacked autoencoder containing encoder and decoder
"""

import os
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

MAX_BYTE = 255.0
VALIDATION_SIZE = 5000
NUM_OUTER_HIDDEN_NEURONS = 100
NUM_INNER_HIDDEN_NEURONS = 30
ACTIVATION = "selu"
N_PIXELS = 28 * 28
CURR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__))) 

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_valid , x_train = x_train[:VALIDATION_SIZE]/MAX_BYTE, x_train[VALIDATION_SIZE:]/MAX_BYTE
y_valid, y_train = y_train[:VALIDATION_SIZE], y_train[VALIDATION_SIZE:]
x_test = x_test / MAX_BYTE



encoder = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)), # image size 
        keras.layers.Dense(units=NUM_OUTER_HIDDEN_NEURONS, activation=ACTIVATION),
        keras.layers.Dense(units=NUM_INNER_HIDDEN_NEURONS, activation=ACTIVATION),
    ]
)

decoder = keras.models.Sequential(
    [
        keras.layers.Dense(units=NUM_OUTER_HIDDEN_NEURONS, activation=ACTIVATION),
        keras.layers.Dense(units=N_PIXELS, activation=keras.activations.sigmoid), # 784 multi-label outputs, probability of a pixel being black(i.e. 0) or while (i.e. 1)
        keras.layers.Reshape(target_shape=(28, 28))
    ]
)

stacked_autoencoder_model = keras.models.Sequential(
    [
        encoder,
        decoder 
    ]
)

stacked_autoencoder_model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=1.5))
history = stacked_autoencoder_model.fit(x_train, x_train, epochs=20) # output and target size =  (28, 28)
stacked_autoencoder_model.save( os.path.join(CURR_DIR, __file__[:-3] + ".keras" ))
