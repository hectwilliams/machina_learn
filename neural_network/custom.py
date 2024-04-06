#!/usr/bin/env python 

"""
    Inheritance to build dynamic or custom models

    Usage: custom.py
"""

import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
import sys 
import keras
from keras import ops
import time 

@keras.saving.register_keras_serializable(package="MyLayers")
class WideDeepModel(keras.Model):
    
    def __init__(self, units=30, activation="relu", **kwargs):
        '''
            create neural network layers
        '''
        super().__init__(** kwargs) # args
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)

    def call(self, inputs):
        '''
            connect neural network
        '''
        input_A, input_B = inputs 
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output

current_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)) ) 
epochs = 50
name = "widedeepmodel.keras"
model_output = os.path.join(current_dir, name)
logdir = os.path.join(current_dir, "logs")

def get_tensor_log_id():
    '''
        tf logs
    '''
    date_id = time.strftime("run_%Y_%m_%d__%H_%M_%S") # time to string
    return os.path.join(logdir, date_id)

# dataset
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

validation_set_size = 10000 #test size | train size 60000
x_test = x_test/255.0
x_valid, x_train = x_train[:validation_set_size]/255.0, x_train[validation_set_size:]/255.0
y_valid, y_train = y_train[:validation_set_size], y_train[validation_set_size:]

# model
model = WideDeepModel()
model.compile( loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer="sgd")

# training set
x_train_half_len = int(np.floor(len(x_train) * 0.5 ) )
x_train_array = [x_train[:x_train_half_len], x_train[x_train_half_len:] ]
y_train_array = [y_train[:x_train_half_len], y_train[x_train_half_len:] ]

# validation set partition 
x_validate_half_len = int(np.floor(len(x_valid) * 0.5))
x_validate_array = [x_valid[:x_validate_half_len], x_valid[x_validate_half_len:]] 
y_validate_array = [y_valid[:x_validate_half_len], y_valid[x_validate_half_len:]] 

# test set partituib
x_test_half_len = int(np.floor(len(x_test) * 0.5))
x_test_array = [x_test[:x_test_half_len], x_test[x_test_half_len:]] 
y_test_array = [y_test[:x_test_half_len], y_test[x_test_half_len:]] 

# callbacks
batch_en_cb = keras.callbacks.LambdaCallback(on_batch_begin=lambda batch,logs : ()) # called at start of each batch
checkpoint_cb = keras.callbacks.ModelCheckpoint(model_output, save_best_only=True) # save checkpoints(best one is kept if crash occurs) 
early_stoppage_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
tensorflow_instance_log_cb = keras.callbacks.TensorBoard(log_dir=get_tensor_log_id())

# estimates
model.fit( x_train_array, y_train_array, epochs=epochs, callbacks= [batch_en_cb, checkpoint_cb, tensorflow_instance_log_cb] , validation_data=(x_validate_array, y_validate_array)) 
evaluation = model.evaluate(x_test_array, y_test_array)

# saves model (API and other connections can use model)
model.save(model_output)

# mimic a service loading model for personal use 
service_conn_model = keras.models.load_model(model_output) # rollback to best model 




