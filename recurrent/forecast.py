"""File contains examples of processing RNN
"""


import os
import numpy as np 
from timeit import default_timer
import keras 
import pandas as pd 
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

BATCH_SIZE = 10000
NUM_STEPS = 50
LOSS = keras.losses.mean_squared_error
OPT = keras.optimizers.Adam()
EPOCHS = 50
CURR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__))) 
rng = np.random.default_rng(seed=42)
early_stopping = EarlyStopping(monitor='loss', min_delta=0.0001, patience=20)

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

# if __name__ == "__main__":
started = default_timer()
data = generate_time_series(batch_size=BATCH_SIZE, num_steps=NUM_STEPS + 1) 
x_train, y_train = data[:7000, :NUM_STEPS], data[:7000, -1]
x_valid, y_valid = data[7000:9000, :NUM_STEPS], data[7000:9000, -1]
x_test, y_test = data[9000:, :NUM_STEPS], data[9000:, -1]

# predict last value in series 
y_pred = x_valid[:, -1] # last sample in known time series
mean_error = np.mean(keras.losses.mean_squared_error( y_true=y_valid, y_pred=y_pred))
# print(f'NATIVE FORCAST- MEAN_SQUARE_ERROR - {mean_error:.4f}')

# SIMPLE NETWORK
model = keras.models.Sequential(
    [
        keras.layers.Flatten( input_shape=(NUM_STEPS, 1) ),
        keras.layers.Dense(1)
    ]
)

# Train
model.compile(loss=LOSS, optimizer=keras.optimizers.Adam())
model.fit(x_train, y_train, epochs=EPOCHS, callbacks=[early_stopping])

# Evaluate
model.evaluate(x_valid, y_valid)

# SIMPLE RECURRENT NETWORK

model_simple_rnn = keras.models.Sequential(
    [
        keras.layers.SimpleRNN(units=1, input_shape=[None, 1], return_sequences= False  ),
    ]
)
model_simple_rnn.compile(loss=LOSS, optimizer=keras.optimizers.Adam())
history = model_simple_rnn.fit(x_train, y_train, epochs=EPOCHS, callbacks=[early_stopping])
model_simple_rnn.evaluate(x_valid, y_valid)

some_time_series = x_valid[0] # (50, 1)
some_time_series = some_time_series[np.newaxis, ...] # (1, 50, 1)
y_predict = model_simple_rnn.predict(some_time_series)

print(f' next predicted price  : {y_predict} ')
print(f' next "actual" price  : {y_valid[0]} ')
plt.figure(1)
plt.plot(x_valid[0], label="raw")
plt.plot( [NUM_STEPS + 1], y_predict, label="model", marker="X", c="red")
plt.plot( [NUM_STEPS + 1], y_valid[0], label="actual", marker="X", c="blue")
plt.legend()
print(model.summary)
pd.DataFrame(history.history).plot() 
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.savefig(  os.path.join(CURR_DIR, __file__ [:-3] + "_history" + '.png')   )
plt.show()
print( f'ELASPED RUN TIME: \t{default_timer() - started}\n' )
