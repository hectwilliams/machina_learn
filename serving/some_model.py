"""Saves a model in SavedModels format"""

import os 
import keras 
import tensorflow as tf
import numpy as np 
import sys
import matplotlib.pyplot as plt

CURR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__))) 
MODEL_PATH = os.path.abspath ( os.path.join(CURR_DIR, "../autoencoder"))
sys.path.append(MODEL_PATH)
import stacked_ae

CURR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__))) 
VERSION_A = "0001"
VERSION_B = "0002"
MODEL_NAME = "my_model"
MODEL_PATH = os.path.join(CURR_DIR, MODEL_NAME, VERSION_A)
LOSS = "binary_crossentropy"
METRICS = [keras.metrics.Accuracy()] 
OPTIMIZER = keras.optimizers.SGD(learning_rate=1.5)
X_NEW = stacked_ae.x_valid 
X_NEW_TRAIN = stacked_ae.x_train



# predict 
x = X_NEW[0]
x_new_local = X_NEW[0:3]

if __name__ == "__main__":
    
    if not os.path.isdir(MODEL_PATH):
        
        np.save( os.path.join(  CURR_DIR , "my_tests.npy"), x_new_local)
        imported = tf.saved_model.load(MODEL_PATH)
        model = imported.signatures["serving_default"]
        pred = model(  tf.constant (x,  dtype=tf.float32) )
        pred = pred['output_0'].numpy()
        plt.imshow(pred[0], cmap="binary")
        plt.savefig(os.path.join(CURR_DIR, __file__ [:-3] + '_visualize_' +'.png'))

    else: 

        model = keras.models.clone_model (stacked_ae.stacked_autoencoder_model)
        model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=[keras.metrics.BinaryAccuracy()] )
        model.fit(X_NEW_TRAIN, X_NEW_TRAIN, epochs=1)
        model.evaluate(stacked_ae.x_valid, stacked_ae.x_valid)
        # print(model.get_weights())

        # for s in model.layers:
        #     for l in s.layers:
        #         print(l)
        #         if isinstance(l, keras.layers.Dense) :
        #             l.trainable = False 

        model.summary()
        tf.saved_model.save(
            model, 
            MODEL_PATH, 
        )
        # model.save(MODEL_PATH)