"""
    Simple CNN, working on mnist fashion dataset 
"""
import os
import keras
import tensorflow as tf 

SOFTMAX_ACTIVATION = "softmax"
RELU_ACTIVATION = "relu"
OPTIMIZER= "sgd" # keras.optimizers.SGD(learning_rate=0.001,momentum= 0.5) 
MODEL_LOSS_= "sparse_categorical_crossentropy"
MODEL_METRICS = ["accuracy"]
PADDING= "same"
CURR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__))) 
VALIDATION_SIZE = 5000
MAX_BYTE = 255.0
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_valid , x_train = x_train[:VALIDATION_SIZE]/MAX_BYTE, x_train[VALIDATION_SIZE:]/MAX_BYTE
y_valid, y_train = y_train[:VALIDATION_SIZE], y_train[VALIDATION_SIZE:]
x_test = x_test / MAX_BYTE

model = keras.models.Sequential(
    [
        keras.layers.Input(shape=[28, 28, 1]),

        keras.layers.Conv2D(filters=64, kernel_size=7, activation=RELU_ACTIVATION, padding=PADDING),

        keras.layers.MaxPooling2D(pool_size=(2,2)), # kernel of max-pool = 2 x 2

        keras.layers.Conv2D(filters=128, kernel_size=3, activation=RELU_ACTIVATION, padding=PADDING),
        
        keras.layers.Conv2D(filters=128, kernel_size=3, activation=RELU_ACTIVATION, padding=PADDING),
        
        keras.layers.MaxPooling2D(pool_size=(2,2)), 

        keras.layers.Conv2D(filters=256, kernel_size=3, activation=RELU_ACTIVATION, padding=PADDING),

        keras.layers.Conv2D(filters=256, kernel_size=3, activation=RELU_ACTIVATION, padding=PADDING),
        
        keras.layers.MaxPooling2D(pool_size=(2,2)),

        keras.layers.Flatten(),

        keras.layers.Dense(128, activation=RELU_ACTIVATION),

        keras.layers.Dropout(0.5),

        keras.layers.Dense(64, activation=RELU_ACTIVATION),

        keras.layers.Dropout(0.5),

        keras.layers.Dense(10, activation=SOFTMAX_ACTIVATION)
    ]
)
keras.utils.plot_model(model, os.path.join(CURR_DIR, __file__[:-3] + '.png'), show_shapes=True)

model.compile(loss=MODEL_LOSS_, optimizer=OPTIMIZER, metrics=MODEL_METRICS)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size=32, drop_remainder=True, num_parallel_calls=3).prefetch(1)
validate_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size=32, drop_remainder=True, num_parallel_calls=3).prefetch(1)

model.fit(train_dataset, validation_data=validate_dataset)

# model.fit(x_train, y_train)

model.save( os.path.join(CURR_DIR, __file__[:-3] + ".keras" ))
