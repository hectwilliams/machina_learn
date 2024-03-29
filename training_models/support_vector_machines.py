from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.lines import Line2D
import threading
import time 

data_lock = threading.Lock() 
count_lock = threading.Lock() 
plt_lock = threading.Lock() 
count_done = threading.Lock()

COLUMNS = 100

def thr_function(data):
    
    global count 
    global d 

    for x2 in data['x2']:

        data_lock.acquire()

        model= data['model']

        x1 = data['x']

        y_predict = model.predict(np.c_[[x1], [x2]])[0]

        id_color = int(y_predict)

        data['plt'].scatter(x1,x2,marker="s", s=5 , c=colors[id_color], alpha=0.1 )

        if plt_lock.locked():
            plt_lock.release()

        data_lock.release()
    
    count_lock.acquire()
    
    count = count + 1
    
    count_lock.release()
    

count_done.acquire()
plt_lock.acquire()
d = {}
count = 0

n_samples = 100

rng = np.random.default_rng()

x, y = make_moons(n_samples=n_samples, noise=0.15)

x1_max = 2
x2_max = 1.5

x_new = 2 * rng.standard_normal(size=(COLUMNS,2))

poly_svm_clf = Pipeline(
    [
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge"))
    ]
)

poly_svm_clf.fit(x, y)

y_predict = poly_svm_clf.predict(x_new) # svm do not output probability


# PLOT CALL ON MAIN THREAD

markers=["s", "^"]
colors = ['tomato', "green",]
labels = ["0", "1"] 

plt.figure()
plt.ylabel ('X\u2082') #  superscript \u00b  subscript \u208
plt.xlabel ('X\u2081')
plt.legend( [ Line2D([0], [0], color="w", markerfacecolor=colors[0],  marker=markers[0]),  Line2D([0], [0], color="w", markerfacecolor=colors[1],   marker=markers[1]),] , labels,  bbox_to_anchor=(1,1)  )
plt.ion()
plt.show()
plt.ylim((-x2_max, x2_max))
plt.xlim((-x1_max, x1_max))

# PLOT CLASSIFICATION MARKERS

for x1, x2, id in np.c_[x_new, y_predict]:
    id = int(id)
    plt.scatter(x1,x2, marker=markers[id], c=colors[id])

x1_space = np.linspace(start= -x1_max, stop=x1_max, num=COLUMNS) # x
x2_space = np.linspace(start= -x2_max, stop=x2_max, num=COLUMNS) # y

data = list(map(lambda x1_point: {'model': poly_svm_clf,'x': x1_point,  'x2': x2_space, 'plt': plt,'colors': colors} , x1_space   ))

# Thread 

for data_obj in data:
    t = threading.Thread( target=thr_function, args=[data_obj] )
    t.start()

# plt.draw and plt.pause must be called on main thread 

while count_done.locked() :
    
    if count == COLUMNS:
        count_done.release()

    plt_lock.acquire()
    plt.draw()
    plt.pause(0.001)

t.join()

plt.ioff()
plt.show()
