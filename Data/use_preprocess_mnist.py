"""Fetch preprocessed datasets and run thru keras DNN
"""

import os 
import keras 
import preprocess.preprocess_mnist
import sys 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

CURR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__))) 
SOFTMAX_ACTIVATION = "softmax"
RELU_ACTIVATION = "relu"
OPTIMIZER= "sgd"
MODEL_LOSS_= "sparse_categorical_crossentropy"
MODEL_METRICS = ["accuracy"]

rng = np.random.default_rng(seed=42)

pre_process_mnist = preprocess.preprocess_mnist.PreprocessMnist(load_tf_records=True)
(train_x, train_y), (test_x, test_y), (valid_x, valid_y) = pre_process_mnist.load_data()

length_percent_25_train = int(0.25 * len(train_x)) 
rand_indices = rng.integers(low =1, high=len(train_x), size=(length_percent_25_train))
adapt_values = train_x[rand_indices]

normalization = keras.layers.Normalization()
normalization.adapt(adapt_values)
normalized_train_x = normalization(train_x)

model = keras.models.Sequential(
            [
                keras.layers.Input(shape=[28,28]),
                keras.layers.Flatten(),
                keras.layers.Dense(300, activation=RELU_ACTIVATION),
                keras.layers.Dense(200, activation=RELU_ACTIVATION),
                keras.layers.Dense(100, activation=RELU_ACTIVATION),
                keras.layers.Dense(10, activation=SOFTMAX_ACTIVATION)
            ]
        )

keras.utils.plot_model(model, os.path.join(CURR_DIR, __file__ [:-3]+ '.png'), show_shapes=True)
model.compile(loss=MODEL_LOSS_, optimizer=OPTIMIZER, metrics=MODEL_METRICS)
history = model.fit(normalized_train_x, train_y, validation_data=(valid_x, valid_y), epochs=30)
print(model.summary())
model.evaluate(test_x, test_y)

pd.DataFrame(history.history).plot() 
plt.grid(True)
plt.gca().set_ylim(0,1)



