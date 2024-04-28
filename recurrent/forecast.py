"""File contains examples of processing RNN
"""

import os
import numpy as np 
from timeit import default_timer
import keras 
import pandas as pd 
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import sys 

BATCH_SIZE = 10000 # 10000
NUM_STEPS = 50
LOSS = keras.losses.mean_squared_error
OPT = keras.optimizers.Adam()
EPOCHS = 1
CURR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__))) 
PREDICT_NUM_NEXT_VALUES = 4

rng = np.random.default_rng(seed=42)
early_stopping = EarlyStopping(monitor='loss', min_delta=0.0001, patience=7)

def generate_time_series(batch_size, num_steps):
    """Generate batch of time series noisy sin waves
    
    Args:
        batch_size: Batched size passed to network
        num_steps: Time series steps 
        
        
    Returns:
        Numpy n-dimensional array of time series data
    """
    freq1, freq2, offsets1, offsets2 = rng.random(size=(4, batch_size, 1))
    time = np.linspace(start=0, stop=1, num=num_steps)
    series = 0.5 * np.sin( ( time - offsets1)* freq1 * 10 ) 
    series += 0.2 * np.sin( ( time - offsets2)* freq2 * 20 ) 
    series += 0.1 * (rng.random(size=(batch_size, num_steps)) - 0.5 )
    return series[..., np.newaxis].astype(np.float32) 

def neural_network(num_steps):
    """Generates a simple neural network to mimic RNN
    

    Args:
        num_steps: 
            Number of time steps in time series 


    Returns:
        Neural Network which accepts time series data
    """
    model = keras.models.Sequential(
        [
            keras.layers.Flatten( input_shape=(num_steps, 1) ),
            keras.layers.Dense(1)
        ]
    )
    model.compile(loss=LOSS, optimizer=keras.optimizers.Adam())
    return model 

def simple_recurrent_network():
    """Generates a simple recurrent network 
    
    
    Returns:
        Neural network containing single RNN layer
    """
    model_simple_rnn = keras.models.Sequential(
        [
            keras.layers.SimpleRNN(units=1, input_shape=[None, 1], return_sequences= False  ),
        ]
    )
    model_simple_rnn.compile(loss=LOSS, optimizer=keras.optimizers.Adam())
    return model_simple_rnn

def deep_recurrent_network(num_next_values_to_predict = 1):
    """Generates a deep recurrent network 
    
    
    Returns:
        Neural network containing more than one RNN layer
    """
    model_deep_rnn = keras.models.Sequential(
        [
            keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
            keras.layers.SimpleRNN(20),
            keras.layers.Dense(num_next_values_to_predict)
        ]
    )
    model_deep_rnn.compile(loss=LOSS, optimizer=keras.optimizers.Adam())
    return model_deep_rnn

def train_it(model, train_x, train_y, valid_x, valid_y):
    """Train a model and plot the forecast prediction
    
    
    Args:
        model: Keras model
        train_x: Training set data
        train_y: Training set labels 
        valid_x: Validation set data
        valid_y: Validation set labels 
    """
    model.compile(loss=LOSS, optimizer=keras.optimizers.Adam())
    model.fit(train_x, train_y, epochs=EPOCHS, callbacks=[early_stopping])
    rand_index = rng.integers(low = 0, high=2000)
    print(train_x.shape)
    print(x_valid.shape)
    print(x_test.shape)
    model.evaluate(valid_x, valid_y)

    some_time_series = x_valid[rand_index] # (50, 1)
    some_time_series = some_time_series[np.newaxis, ...] # (1, 50, 1)
    y_predict = model.predict(some_time_series)
    print(y_predict)
    y_label = y_valid[rand_index]

    print(some_time_series[0].shape)
    print(y_predict.reshape((4,1)).shape )
    
    plt.figure(1)
    plt.plot(  np.vstack ( (some_time_series[0]  , y_label[..., np.newaxis] ) )  , label="raw", c="pink")
    plt.plot( np.array(range(NUM_STEPS, NUM_STEPS + PREDICT_NUM_NEXT_VALUES )).reshape((1,PREDICT_NUM_NEXT_VALUES)) , y_predict, marker="X", c="blue", markersize=2)
    plt.plot(  np.vstack ( (some_time_series[0]  , y_predict.reshape((4,1))  ) )  , label="predicts", c="green")
    plt.legend()
    plt.show()

data = generate_time_series(batch_size=BATCH_SIZE, num_steps= NUM_STEPS + PREDICT_NUM_NEXT_VALUES) 
x_train, y_train = data[:7000, :NUM_STEPS], data[:7000, -PREDICT_NUM_NEXT_VALUES:, 0]
x_valid, y_valid = data[7000:9000, :NUM_STEPS], data[7000:9000, -PREDICT_NUM_NEXT_VALUES:, 0]
x_test, y_test = data[9000:, :NUM_STEPS], data[9000:, -PREDICT_NUM_NEXT_VALUES:, 0]
model = deep_recurrent_network(PREDICT_NUM_NEXT_VALUES) 
train_it(model=model, train_x=x_train, train_y=y_train, valid_x=x_valid, valid_y=y_valid)

