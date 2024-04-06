import os
import pickle
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt 
from tensorflow import keras

current_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)) ) 
fashion_mnist = keras.datasets.fashion_mnist
epochs = 1

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train.shape # (60000, 28, 28) 
y_train.shape # (60000,) 
x_test.shape, # (10000, 28, 28) 
y_test.shape  # (10000,)

# create cross validation set 
validation_set_size = 5000
x_valid, x_train = x_train[:validation_set_size], x_train[validation_set_size:]
y_valid, y_train = y_train[:validation_set_size], y_train[validation_set_size:]

# normalize (scale) pixel intensities 
x_valid = x_valid / 255.0
x_train = x_train / 255.0
x_test  = x_test / 255.0

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

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28))) # [28, 28] image, flattened to [1, 784]
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax")) # 10 output probabilities (sum of probabilities = 1)

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_valid, y_valid))

info = model.evaluate(x_test, y_test) #[ test loss, test accuracy ]

# plot training metrics 
tf.keras.utils.plot_model(model, to_file=os.path.join(current_dir, "model_plot.png"), show_shapes=True, show_layer_names=True)
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical rrange to [ 0 - 1 ]

# save 
file_output = os.path.join(current_dir, "mlp.h5")
model.save(file_output)

plt.show() 