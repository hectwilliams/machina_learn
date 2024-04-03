import os 
import sys 
import numpy as np
from matplotlib.image import imread
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt 

wk_dir = os.path.join(os.path.dirname(os.path.realpath(__file__))) 

def get_mnist(tr_size=10000):

    mnist = fetch_openml('mnist_784', version=1)

    X, y = mnist['data'], mnist['target']
    y =  y.astype(np.uint8) # cast string category type (i.e. '1', '2' ) to integers (i.e. 1 , 2 )
    return X[:tr_size], X[tr_size:], y[:tr_size], y[tr_size:] # pre-shuffled dataset 


def ladybug_k_means():

    img = imread(os.path.join(wk_dir, "images", "ladybug.png")) # img.shape (1452, 1452, 3) row, col, rgb

    # reshape to list of rgb data 
    x = img.reshape(-1, 3)

    # cluster image into 8 regions
    kmeans = KMeans(n_clusters=8).fit(x)

    # score = silhouette_score(x, kmeans.labels_)

    # assign kmean labels to cluster 
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]

    # reshape segmented or clustered  pixel(rgb pixel) data
    segmented_img = segmented_img.reshape(img.shape)

def digit_cluster():
    
    x_digits, y_digits = load_digits(return_X_y=True)

    x_train, x_test, y_train, y_test = train_test_split(x_digits, y_digits)

    log_reg = LogisticRegression()

    log_reg.fit(x_train, y_train)

    score = log_reg.score(x_test, y_test)

    # find optimal k 
    
    pipeline = Pipeline(
        [
            ("kmeans", KMeans()),
            ("std_scaler", StandardScaler()),
            ("log_reg", LogisticRegression())
        ]
    )

    param_grid = dict(kmeans__n_clusters=range(2, 100))
    grid_clf = GridSearchCV(pipeline, param_grid=param_grid, cv=3)
    grid_clf.fit(x_train, y_train)

    print("accuracy before kmeans segmentation: {} \t after {} ".format( score, grid_clf.score(x_test, y_test) ))

def semi_supervise():

    n_labeled = 50

    x_train, x_test, y_train, y_test = get_mnist(n_labeled)

    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)
    
    score_before_kmeans = log_reg.score(x_test, y_test)

    k = 50
    kmeans = KMeans(n_clusters=k)
    x_digits_dist = kmeans.fit_transform(x_train) # measures distance from each instance to every centroid (50 samples x 50 distance-deltas from nth sample)

    # kmeans.cluster_centers_ 50 clusters at 50 points of 784 coordinates 

    representative_digit_idx = np.argmin(x_digits_dist, axis=0) # closest instance to the 50 centroids 

    # plot cluster representatives 
    _, axes = plt.subplots(ncols=10, nrows=5)

    for i in range(n_labeled):
        
        rows = i // 10 
        cols = i % 10

        ax = axes[rows, cols]

        representative_label = representative_digit_idx[i]

        ax.imshow(x_train.loc[representative_label].to_numpy().reshape(28, 28) , cmap='binary')

        # remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # remove border 
        ax.set_frame_on(False)
    
    plt.show() 

semi_supervise()
