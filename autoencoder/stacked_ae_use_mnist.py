"""Test mnist dataset on stacked autoencoder"""
import os
import numpy as np 
import sys
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.image import imread 

CURR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__))) 
MAX_BYTE = 255.0
VALIDATION_SIZE = 5000
N_PIXELS = 28 * 28

def plot_image(image):
    """Plots image and remove axis
    
    
    Args:
        image: _
    """
    plt.imshow(image, cmap="binary")
    plt.axis("off") 

def view_reconstruction(model, n_images):
    """Plots model's reconstructed images 
    
    
    Args:
        model: Keras model containing autoencoder
        n_images: Number of original and reconstruction sets to show
    """
    global x_valid
    reconstructed_images = model.predict(x_valid[:n_images])
    fig = plt.figure(figsize=(n_images, 3)) # width height inches 
    for i in range(n_images):
        plt.subplot(2, n_images, i + 1)
        plot_image(x_valid[i])
        plt.subplot(2, n_images, n_images + i + 1)
        plot_image(reconstructed_images[i])

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_valid , x_train = x_train[:VALIDATION_SIZE]/MAX_BYTE, x_train[VALIDATION_SIZE:]/MAX_BYTE
y_valid, y_train = y_train[:VALIDATION_SIZE], y_train[VALIDATION_SIZE:]
x_test = x_test / MAX_BYTE
model = keras.models.load_model(os.path.join(CURR_DIR, "stacked_ae.keras") )
view_reconstruction(model, 5)
plt.savefig(  os.path.join(CURR_DIR, __file__ [:-3] + '.png')   )