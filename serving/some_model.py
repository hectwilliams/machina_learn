"""Saves a model in SavedModels format"""

import os 
import tensorflow_datasets as tfds
import keras 
import sys 
import tensorflow as tf
import numpy as np 
import sys

CURR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__))) 
MODEL_PATH = os.path.abspath ( os.path.join(CURR_DIR, "../autoencoder"))
sys.path.append(MODEL_PATH)
import stacked_ae

QUICK_PREDICT = False 
VERSION_A = "0001"
VERSION_B = "0002"
MODEL_NAME = "my_model"
MODEL_PATH = os.path.join(CURR_DIR, MODEL_NAME, VERSION_A)
LOSS = "binary_crossentropy"
METRICS = [keras.metrics.Accuracy()] 
OPTIMIZER = keras.optimizers.SGD(learning_rate=0.001) 
X_NEW = stacked_ae.x_valid 
X_NEW_TRAIN = stacked_ae.x_train

# np.save( os.path.join(CURR_DIR , "my_tests.npy") , X_NEW )

if __name__ == "__main__":
    model = stacked_ae.stacked_autoencoder_model
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    model.fit(X_NEW_TRAIN, X_NEW_TRAIN, epochs=5)
    print(model.get_weights())
    tf.saved_model.save(model, MODEL_PATH)