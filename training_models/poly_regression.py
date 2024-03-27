import sys 
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import model_metrics
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler

def plot_simple(x, y, y_pred):
    # plot 
    plt.figure()
    plt.ylabel("y")
    plt.xlabel("x")
    plt.title("poly regress")
    plt.scatter(x=x, y=y, c='b')
    plt.scatter(x, y_pred,c="red",linewidth=0.4,marker=".")
    plt.legend( [ Line2D([0], [0],   markerfacecolor='b', color='w', markersize=10, marker='o', linestyle='-'), Line2D([0], [0], color="red", linewidth=3, linestyle='-')] , ["noisy quadratic", "prediction"])
    plt.show()

rng = np.random.default_rng()

m = 500
x = 6 * rng.random(size=(m,1)) - 3
# x.sort(axis=0) # sort random variables 

noise_normal_dist = rng.standard_normal(size=(m,1)) 
y = 0.5 * x**2 + x + 2 + noise_normal_dist

# transformed data 
poly_features = PolynomialFeatures(degree= 2, include_bias=False)
x_poly = poly_features.fit_transform(x)
x_b_poly = np.c_[np.ones(shape=(m,1)), x_poly] # bias included

# estimator
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)

# weights 
weights = np.c_[lin_reg.intercept_, lin_reg.coef_]

# prediction  X^2 * w_2 + X * w_1 + w_0
y_pred = lin_reg.predict(x_poly)   # estimator compute
# y_pred = x_b_poly.dot(weights.T) # matrix compute

# plot
plot_simple(x, y, y_pred)

# learning curve
model_metrics.learning_curve(lin_reg, x, y)

# 
polynomial_reg = Pipeline([
    # ('std_scaler', StandardScaler()),
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression())
])

model_metrics.learning_curve(polynomial_reg, x, y)
