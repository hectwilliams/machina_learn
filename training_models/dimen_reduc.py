
from sklearn.datasets import fetch_openml
import os 
import sys
import numpy as np 

from sklearn.decomposition import PCA 
from matplotlib import pyplot as plt, colormaps

rng = np.random.default_rng()

def get_mnist(tr_size=30000):

    mnist = fetch_openml('mnist_784', version=1)

    X, y = mnist['data'], mnist['target']

    y =  y.astype(np.uint8) # cast string category type (i.e. '1', '2' ) to integers (i.e. 1 , 2 )

    return X[:tr_size], X[tr_size:], y[:tr_size], y[tr_size:] # pre-shuffled dataset 

# training data 

def mnist_compression_example():
    x_train, x_test, y_train, y_test = get_mnist()
    # find right number of dimensions 
    pca = PCA() 
    pca.fit(x_train)

    # plot percent variance 
    plt.figure(1)
    plt.plot(pca.explained_variance_ratio_ )
    plt.ylabel('percent of variance')
    plt.xlabel('principle component')

    # find number of components needed to meet a 95 percent variance 
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    index_95_cumsum = np.argmax(cumsum >= 0.95) 
    dimensions = index_95_cumsum + 1
    dimensions = np.ceil(np.sqrt(dimensions)) ** 2 
    dimensions_sqrt = int(np.sqrt(dimensions))
    pca = PCA(n_components= dimensions_sqrt**2)
    x2d = pca.fit_transform(x_train) # compressed data 
    some_digit = x2d[0].reshape(dimensions_sqrt,dimensions_sqrt)

    fig, axes = plt.subplots(nrows= 1, ncols=2 )

    # show compressed image
    axes[0].imshow(some_digit, cmap= colormaps["binary"])
    axes[0].set_title('Compressed Image')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # recover the data 
    x_recovered = pca.inverse_transform(x2d)
    some_digit_recovered = x_recovered[0].reshape(28, 28)
    axes[1].imshow(some_digit_recovered, cmap= colormaps["binary"])
    axes[1].set_title('Decompressed Image')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.show()

def test_me():

    f = 10
    Fs = 1000
    n = 10000
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    z = rng.lognormal(size=(n,1)) 
    y =  rng.standard_normal(size=(n,1)) # rng.random() * np.cos(2 * np.pi *f * z / Fs)
    x =  rng.standard_normal(size=(n,1)) #   np.sin(2 * np.pi *f * z / Fs)

    ax.scatter(xs=x,ys=z, zs=y)
    # ax.scatter(xs=np.zeros(n) - 1,ys=z, zs=y)
    # ax.scatter(xs=x,ys=z, zs=np.zeros(n) -1)

    ax.set_xlabel('X Label')
    ax.set_ylabel('time')
    ax.set_zlabel('Z Label')

    plt.show() 

test_me()