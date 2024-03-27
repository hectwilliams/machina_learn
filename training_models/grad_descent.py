import numpy as np
from llinear_regression import x, x_b, y, m_size
import matplotlib.pyplot as plt 

def get_gradients(x,y, w, size_):
    error = x.dot(w) - y
    row_of_features = x.T # rotate column of features values to rows
    gradients = (2/size_) * row_of_features.dot(error)
    return gradients

rng = np.random.default_rng()

eta = 0.08

iterations = 100

m = m_size # instances 

w = rng.random(size=(2,1)) # random guess (weights or parameters)

if __name__ == "__main__":
    plt.figure(1)
    plt.title("η = {}".format(eta))
    plt.scatter(x=x, y=y, label="raw scores")
    plt.legend()
    plt.xlabel("id")
    plt.ylabel("scores")
    plt.ylim((0, 14))
    plt.xlim((0, 2))

    # plt.show(block=False)
    plt.ion()
    plt.show() 

    curr_mean = 2*32
    cont = True

    for i in range(iterations):
        curr_guess_errors = x_b.dot(w) - y
        
        abs_mean = abs(curr_guess_errors.mean()) 
        
        if curr_mean -  abs_mean   < 1/100:
            cont = False 

        if curr_mean > abs_mean:
            curr_mean = abs_mean

        row_of_features = x_b.T # rotate column of features values to rows
        gradients  = get_gradients(x=x_b, y=y, w=w, size_=m_size)# (2/m) * row_of_features.dot(curr_guess_errors)
        w = w - eta * gradients

        # predict usnig weights
        pred = x_b.dot(w)
        
        # plot prediction (update plot)
        plt.plot(x, pred, c="tan"  )

        # plot draw (queued event) 
        plt.draw() 

        # sleep thread (run queued event)
        plt.pause(0.001)
        
        if not cont:
            print("weights = {}".format(w))
            break 

    plt.ioff()
    plt.show()
    print('Done')



