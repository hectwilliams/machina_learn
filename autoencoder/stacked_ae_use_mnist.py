"""Test mnist dataset on stacked autoencoder"""
import os
import numpy as np 
import sys
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle 
from matplotlib.image import imread 
from sklearn.manifold import TSNE 
from stacked_ae import encoder
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

CURR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__))) 
MAX_BYTE = 255.0
VALIDATION_SIZE = 5000
N_PIXELS = 28 * 28
CLASS_LABELS = [
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
PKL_FILE = os.path.join(CURR_DIR, "stacked_ae.pkl")

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
        
def visualize_mnist(use_local_data=False):
    """Encoder's output (i.e. codings) will undergo
    PCA (via t-SNE), knocking down the dimensionality to 2D to visualize 
    data patterns


    Args:
        use_local_data: Boolean dictating whether function
        will generate dimensional reduction data or use the 
        data stored locally in this directory 
    """
    global x_valid
    global y_valid
    global annots

    table_images = {}
    x_valid_2D = np.empty(shape=(1,1))
    
    if use_local_data:
        with open(PKL_FILE, 'rb') as f:
            obj = pickle.load(f)
            x_valid_2D = obj['data']
            table_images = obj['disp_table']
    else:
        tsne = TSNE()
        x_valid_compressed = encoder(x_valid)
        x_valid_2D = tsne.fit_transform(x_valid_compressed)
        for i in range(len(y_valid)):
            table_images[y_valid[i]] = [ [ x_valid_2D[i, 0], x_valid_2D[i, 1] ] , x_valid[i] ]
        with open(PKL_FILE, "wb") as f:
            pickle.dump( obj={"disp_table": table_images, "data": x_valid_2D }, file=f)
    
    fig, ax = plt.subplots()
    ax.scatter(x_valid_2D[:, 0], x_valid_2D[:, 1], c=y_valid, s= 10, cmap="tab10")

    for key in table_images:
        table_entry = table_images[key]
        img = table_entry[1]
        row = table_entry[0][0]
        col = table_entry[0][1]
        imagebox = OffsetImage(img, zoom=0.5)
        annotation_bbox = AnnotationBbox(imagebox, (col, row), frameon= False)
        ax.add_artist(annotation_bbox)
        annots.append(ax.annotate(CLASS_LABELS[key], xy=(col, row), xytext=(col, row+3 ), fontsize=7, color="black",  weight='bold')) 
    plt.savefig(os.path.join(CURR_DIR, __file__ [:-3] + '_visualize_' +'.png'))
    plt.show()

annots = []
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_valid , x_train = x_train[:VALIDATION_SIZE]/MAX_BYTE, x_train[VALIDATION_SIZE:]/MAX_BYTE
y_valid, y_train = y_train[:VALIDATION_SIZE], y_train[VALIDATION_SIZE:]
x_test = x_test / MAX_BYTE
model = keras.models.load_model(os.path.join(CURR_DIR, "stacked_ae.keras") )
visualize_mnist(True)
# view_reconstruction(model, 5)
# plt.savefig(  os.path.join(CURR_DIR, __file__ [:-3] + '.png')   )