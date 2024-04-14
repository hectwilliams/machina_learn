"""Fetch preprocessed datasets and run thru keras DNN
"""

import os 
import keras 
import preprocess.preprocess_mnist
import pandas as pd 
import matplotlib.pyplot as plt 

CURR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__))) 
SOFTMAX_ACTIVATION = "softmax"
RELU_ACTIVATION = "relu"
OPTIMIZER= "sgd"
MODEL_LOSS_= "sparse_categorical_crossentropy"
MODEL_METRICS = ["accuracy"]

pre_process_mnist = preprocess.preprocess_mnist.PreprocessMnist(load_tf_records=True)
train_feature, train_label = pre_process_mnist.load_dataset(dataset_name='training')
validate_feature, validate_label = pre_process_mnist.load_dataset(dataset_name='validate')
test_feature, test_label = pre_process_mnist.load_dataset(dataset_name='test')

model = keras.models.Sequential(
            [
                keras.layers.Input(shape=[28,28]),
                pre_process_mnist.standardizer,
                keras.layers.Flatten(),
                keras.layers.Dense(300, activation=RELU_ACTIVATION),
                keras.layers.Dense(200, activation=RELU_ACTIVATION),
                keras.layers.Dense(100, activation=RELU_ACTIVATION),
                keras.layers.Dense(10, activation=SOFTMAX_ACTIVATION)
            ]
        )

keras.utils.plot_model(model, os.path.join(CURR_DIR, __file__ [:-3] + "_flow" + '.png'), show_shapes=True)
model.compile(loss=MODEL_LOSS_, optimizer=OPTIMIZER, metrics=MODEL_METRICS)
history = model.fit(train_feature, train_label  ,epochs=5, validation_data=(validate_feature, validate_label))
model.evaluate(test_feature, test_label)

print(model.summary())
pd.DataFrame(history.history).plot() 
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.savefig(  os.path.join(CURR_DIR, __file__ [:-3] + "_history" + '.png')   )