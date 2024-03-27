import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors

def plot_example(x,y, y_predict):
    plt.figure()
    plt.scatter(x=x, y=y, label="raw scores")
    plt.plot(x, y_predict,c="tan", label='model')
    plt.legend()
    plt.xlabel("id")
    plt.ylabel("scores")
    plt.grid(visible="major")
    plt.title("Linear Regression Model")
    plt.show()

m_size = 100
rng = np.random.default_rng(seed=42)

x = 2 * rng.random(size=(m_size, 1))
y = 4 + 3 * x + rng.standard_normal(size=(m_size, 1))

# Closed form Normal Equation, weights that minimize MSE
x_b = np.c_[np.ones(shape=(m_size,1)), x]
weights_minimized = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)  # w0 + w1

y_predict = x_b.dot(weights_minimized)

if __name__ == "__main__" :
    plot_example(x, y, y_predict)
   