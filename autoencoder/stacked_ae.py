""""Creates a stacked autoencoder containing encoder and decoder
"""

import os
import tensorflow as tf
import keras 
import matplotlib.pyplot as plt

MAX_BYTE = 255.0
VALIDATION_SIZE = 5000
NUM_OUTER_HIDDEN_NEURONS = 100
NUM_INNER_HIDDEN_NEURONS = 30
ACTIVATION = "selu"
N_PIXELS = 28 * 28
CURR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__))) 
USE_SEQ = True

def func_model():
    encoder = keras.layers.Input(shape=(28, 28))
    enc_flatten = keras.layers.Flatten() (encoder)
    enc_dense0 = keras.layers.Dense(units=NUM_OUTER_HIDDEN_NEURONS,  activation="selu",  kernel_initializer='random_normal', bias_initializer='zeros') (enc_flatten)
    enc_dense1 = keras.layers.Dense(units=NUM_OUTER_HIDDEN_NEURONS, activation="selu", kernel_initializer='random_normal', bias_initializer='zeros') (enc_dense0)

    dec_dense0 = keras.layers.Dense(units=NUM_OUTER_HIDDEN_NEURONS, activation="selu", kernel_initializer='random_normal', bias_initializer='zeros') (enc_dense1)
    dec_dense1 = keras.layers.Dense(units=N_PIXELS, activation=keras.activations.sigmoid) (dec_dense0)
    decoder  = keras.layers.Reshape(target_shape=(28, 28)) (dec_dense1)

    return keras.models.Model(inputs=[encoder], outputs=[decoder])

def seq_model():
    encoder = keras.Sequential(
        [
            keras.layers.Input(shape=(28, 28)),
            keras.layers.Flatten() ,
            keras.layers.Dense(units=NUM_OUTER_HIDDEN_NEURONS, activation="selu", use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros', name="d1"), 
            keras.layers.Dense(units=NUM_INNER_HIDDEN_NEURONS, activation="selu", use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros', name="d2") ,
        ]
    )
    decoder = keras.Sequential(
        [
            keras.layers.Dense(units=NUM_OUTER_HIDDEN_NEURONS, activation="selu", use_bias=True, input_shape=[30], kernel_initializer='random_normal', bias_initializer='zeros', name="d3" ), 
            keras.layers.Dense(28 * 28, activation="sigmoid", use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros',  name="d4"),
            keras.layers.Reshape(target_shape=(28, 28))
        ]
    )
    return keras.models.Sequential( [encoder, decoder] )

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_valid , x_train = x_train[:VALIDATION_SIZE]/MAX_BYTE, x_train[VALIDATION_SIZE:]/MAX_BYTE
y_valid, y_train = y_train[:VALIDATION_SIZE], y_train[VALIDATION_SIZE:]
x_test = x_test / MAX_BYTE

stacked_autoencoder_model = seq_model() if USE_SEQ else func_model()

if __name__ == "__main__":
    stacked_autoencoder_model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=1.5))
    history = stacked_autoencoder_model.fit(x_train, x_train, epochs=20) # output and target size =  (28, 28)
    keras.utils.plot_model(stacked_autoencoder_model, os.path.join(CURR_DIR, f'{__file__[:-3]}' + "flow" + ".png"), show_shapes=True)
    stacked_autoencoder_model.save( os.path.join(CURR_DIR, __file__[:-3] + ".keras" ))
