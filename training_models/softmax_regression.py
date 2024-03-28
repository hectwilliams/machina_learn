import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from matplotlib.lines import Line2D
import softmax_helper
import threading
import sys 

rng = np.random.default_rng()

iris = datasets.load_iris()

# print(iris.keys()) # ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']

# print(iris['target_names']) # ['setosa' 'versicolor' 'virginica']

data = iris['data'] # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = data[:, [2,3]]

x_petal_length = data[:, [2]]
x_petal_width = data[:, [3]]

# print(x_2, x_3)

y = iris['target']

# create softmax estimator 

# hyperparameters enable Softmax Regression support 
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=5)
softmax_reg.fit(np.c_[x_petal_width, x_petal_length], y)

n = 500
max_petal_length = 7
max_petal_width = 3

x_new_length = max_petal_length * rng.standard_normal(size=(n, 1))
x_new_width = max_petal_width * rng.standard_normal(size=(n, 1))
x_new = np.c_[x_new_width, x_new_length]
y_predict = softmax_reg.predict(x_new)

markers = ['o', "s", "^", "."]   # ['setosa' 'versicolor' 'virginica'
colors = ['tomato', "green", "blue", "silver"]
names = iris['target_names'].tolist() +  ["boundary"]

plt.figure()
plt.ylabel("Petal Width (cm)")
plt.xlabel("Petal Length (cm)")
plt.ylim((0 , max_petal_width))
plt.xlim((0 , max_petal_length))
plt.legend( [ Line2D([0], [0], color="w", markerfacecolor=colors[0],  marker=markers[0]), 
             Line2D([0], [0], color="w", markerfacecolor=colors[1],   marker=markers[1]),
             Line2D([0], [0], color="w", markerfacecolor=colors[2], marker=markers[2]),
             Line2D([0], [0], color="w", markerfacecolor=colors[3], marker=markers[3]),

             ] , names  )

for p_width, p_length, id in np.c_[x_new, y_predict]:
    id = int(round(id))
    plt.scatter( x=p_length, y=p_width, marker=markers[id] , c=colors[id])

bound_search = softmax_helper.Decision_Boundary(max_petal_length, max_petal_width, softmax_reg, plt)
bound_search.search_large()

plt.show()
