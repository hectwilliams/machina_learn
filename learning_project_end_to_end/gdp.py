import matplotlib.pyplot as plt 

import numpy as np

import pandas as pd 

import sklearn.linear_model

# load data 
gdp = pd.read_csv("Data/gdp.csv", thousands=",")

# country 
country = gdp["Country Name"].unique()

# country code 
country_code = gdp["Code"].unique()

# keys 
keys = gdp.keys() 

# gdp usa 
x_years_num = range(1960, 2021)
gdp_usa = gdp.loc[   (gdp["Code"] == 'USA' ),   [ str(k) for k in x_years_num ]   ]

# train set data 
x_train = np.c_[x_years_num]

# train set target 
y_train = np.c_[(gdp_usa.to_numpy() / 1e12).flatten()]

# linear regression linear model 
model = sklearn.linear_model.LinearRegression()

# train the model 
model.fit(x_train, y_train)

# make a prediction for 2024 
# x_new = [[2024]]

# prediction on training set 
y_predict = model.predict(x_train)

# plot training set model  
plt.ylabel("Trillions $")
plt.title("USA GDP")
plt.xlabel("Year")
plt.scatter(x_train, y_train, s= [5]  )
plt.plot(x_train, y_predict, color='green', linewidth= 1 )

plt.show()

