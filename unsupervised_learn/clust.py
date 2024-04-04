import os 
import sys 
import numpy as np
from matplotlib.image import imread
from sklearn.cluster import KMeans , DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt ,colormaps 
from sklearn.datasets import make_moons
from matplotlib.lines import Line2D
from  sklearn.neighbors import KNeighborsClassifier
import threading
import time

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

def dbscan_ex():
    n_samples = 1000
    eps = 0.20

    x, y = make_moons(n_samples=n_samples, noise=0.05)

    dbscan = DBSCAN(eps=eps, min_samples=5)

    dbscan.fit(x)

    colors_range = np.arange(len(set(dbscan.labels_)))
    colors_amplitudes = (3/3 * colors_range).astype(int)
    colors = colormaps['tab20'](colors_amplitudes)
    legend_markers = []
    legend_markers_id = colors_range.astype(str)
    
    for color_ in  colors:
        legend_markers.append( Line2D([0], [0],   markerfacecolor=color_, color='w', markersize=10, marker='o', linestyle='-'))

    x_min = -1.5
    x_max = 2.5
    y_min = -1.25
    y_max = 1.5

    plt.figure(1)
    plt.legend(legend_markers, legend_markers_id,  bbox_to_anchor=(1,1) )
    plt.ylim((y_min, y_max))
    plt.xlim((x_min, x_max))
    plt.ion()
    plt.show()

    for idx, xy in enumerate(x):
        x = xy[0]
        y = xy[1]
        c = dbscan.labels_[idx]
        plt.scatter(x, y , c=colors[c])   
    plt.draw()
    plt.pause(0.2)

    # KNN  MODEL

    knn = KNeighborsClassifier(n_neighbors=50)
    knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])

    # 4 random instances ( X marker on plot )
    walk = 1 
    x_quick = np.c_[ [ [-0.5, 0], [0, 0.5], [1, -0.1], [2,1] ] ]

    for x1, x2 in x_quick:
        y_predict = knn.predict([[x1, x2]])
        y_predict = y_predict.astype(int)

        plt.scatter(x=x1, y=x2, marker="x", )
        plt.annotate( 
            f'({walk})',
            xy=(x1, x2), 
            textcoords="offset points",
            xytext=(5, 5),
        )
        walk += 1

    y_dist, y_pred_idx = knn.kneighbors(x_quick, n_neighbors=1)
    y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx ] # return labels of closest instance to each guess 
    y_pred[y_dist > 0.2] = -1 # predicted anamolies are overwritten to -1
    plt.draw()
    plt.pause(0.05)

    # CREATE BOUNDARY
            
    workers = threading.Lock()
    counter_worker = threading.Lock()
    plt_worker = threading.Lock() 
    data_worker = threading.Lock() 

    workers.acquire()
    plt_worker.acquire()
    
    def col_fill_thread(data):

        for x2 in data['row']:

            data_worker.acquire()

            x1 = data["x"]
            model = data['model']        
            plt_obj = data['plt']   

            y_predict = model.predict_proba(np.c_[[x1], [x2]])[0]
            y_predict_bool = np.logical_and( (y_predict > 0.45) , (y_predict < 0.55))

            if True in y_predict_bool:
                plt_obj.scatter(x1, x2, s=10.3, marker=".", c="red", alpha=1.0)

            if plt_worker.locked():
                plt_worker.release()
            
            data_worker.release()
        
        counter_worker.acquire()
        data['counter'][data['id']] = 1
        counter_worker.release()

    c_length = 100
    cols = np.linspace(start = x_min, stop=x_max, num= c_length)
    row = np.linspace(start = y_min, stop=y_max, num= c_length)
    counter_table = {}

    data = list(map(lambda x1_point: {'model': knn,'x': x1_point[1], 'len': len(row) , 'row': (row), 'plt': plt, 'counter': counter_table, 'id': x1_point[0]} , enumerate(cols)   ))
    threads = list()

    for data_obj in data:
        t = threading.Thread( target=col_fill_thread, args=[data_obj] )
        threads.append(t)
        t.start()
    
    while workers.locked():
        
        if not plt_worker.locked():
            plt_worker.acquire()
            plt.draw()
            plt.pause(0.001)
        else:
            plt.draw()
            plt.pause(0.001)
            plt_worker.release()
            workers.release()

    for t in threads:
        t.join()
    
    print("finished")

    plt.ioff()
    plt.show()

dbscan_ex()