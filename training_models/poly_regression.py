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
from sklearn.linear_model import Ridge 

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

m = 100
x = 6 * rng.random(size=(m,1)) - 3

noise_normal_dist = rng.standard_normal(size=(m,1)) 
y = 0.5 * x**2 + x + 2 + noise_normal_dist

# transformed data 
poly_features = PolynomialFeatures(degree= 2, include_bias=False)
x_poly = poly_features.fit_transform(x)
x_b_poly = np.c_[np.ones(shape=(m,1)), x_poly] # bias included

## LINEAR REGRESS

lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)

# weights 
weights = np.c_[lin_reg.intercept_, lin_reg.coef_]

# prediction  X^2 * w_2 + X * w_1 + w_0
y_pred = lin_reg.predict(x_poly)   # estimator compute
# y_pred = x_b_poly.dot(weights.T) # matrix compute

# sort random variables x with depedant values (i.e. y, y_pred)
temp = np.c_[x, y, y_pred]
temp[(  temp[:,0] ).argsort()]
x = np.c_[temp[:,0]]
y = np.c_[temp[:,1]]
y_pred =np.c_[temp[:,2]]

plot_simple(x, y, y_pred)
model_metrics.learning_curve(lin_reg, x, y)

## POLYNOMIAL REGRESS

polynomial_reg = Pipeline([
    # ('std_scaler', StandardScaler()),
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression())
])

polynomial_reg.fit(x_poly, y)
y_pred = polynomial_reg.predict(x_poly)   # estimator compute

plot_simple(x, y, y_pred)
model_metrics.learning_curve(polynomial_reg, x, y)

## Ridge Regression ( reduce model variace)

## LINEAR REGRESS
ridge_reg = Ridge(alpha=1e-05, solver="cholesky")
ridge_reg.fit(x, y)
y_pred = ridge_reg.predict(x)
plot_simple(x, y, y_pred)
model_metrics.learning_curve(ridge_reg, x, y)

## POLYNOMIAL REGRESS
polynomial_reg = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("ridge_reg", Ridge(alpha=1e-05, solver="cholesky"))
])
polynomial_reg.fit(x_poly, y)
y_pred = polynomial_reg.predict(x_poly)  
plot_simple(x, y, y_pred)
