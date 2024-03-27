import sys 
import numpy as np
from llinear_regression import x, x_b, y, m_size
import matplotlib.pyplot as plt 
from grad_descent import get_gradients
from matplotlib.lines import Line2D
from sklearn.linear_model import SGDRegressor

epochs = 50
t0, t1 = 5, 50

def learning_schedule(t):
    return t0 / (t + t1)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

fig.suptitle("η =")
ax.scatter(x=x, y=y,c="blue")
ax.set_xlabel("id")
ax.set_ylabel("scores")
ax.set_ylim((0, 14))
ax.set_xlim((0, 2))
plt.legend( [ Line2D([0], [0],   markerfacecolor='b', color='w', markersize=10, marker='o'), Line2D([0], [0], color="tan", linewidth=3, linestyle='-')] , ["scores", "model"])

def custom_():
    plt.ion()
    plt.show() 


    rng = np.random.default_rng()

    w = rng.random(size=(2,1)) # guess initial weights 
    data = []
    predict_lines = []
    childs = []
    iteration_per_epoch = 500

    for epoch in range(epochs):
        
        for i in range(iteration_per_epoch):

            rand_i = rng.integers(low=0, high=m_size)

            x_i = x_b[[rand_i]]

            y_i = y[[rand_i]]

            gradients = get_gradients(x_i, y_i, w, m_size)

            eta = learning_schedule(epoch * m_size + i)

            w = w - eta * gradients

        pred = x_b.dot(w)
        
        fig.canvas.draw_idle()

        plt.pause(0.001)
        
        predict_lines.append(ax.plot(x, pred, c="tan")[0] )
        
        childs.append(ax.lines)

    while len(ax.get_lines()) > 1:
        ax.lines[0].remove()
        fig.canvas.draw_idle()
        plt.pause(0.001)

    print("weights=\t {}".format(w))
    plt.ioff()
    plt.show()

def sklearn_version():
    sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
    sgd_reg.fit(x, y.ravel())
    calc_weights = np.c_[sgd_reg.intercept_, sgd_reg.coef_]
    y_pred = x_b.dot(calc_weights.T)
    plt.plot(x, y_pred, c="tan")
    plt.show() 

sklearn_version()