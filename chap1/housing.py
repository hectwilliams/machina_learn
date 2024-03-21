import pandas as pd 
import housing_download as house_api
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib import colormaps
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_reg = LinearRegression()
std_scaler = StandardScaler()
ord_encoder = OrdinalEncoder()
one_hot_encoder = OneHotEncoder()
housing = house_api.load_housing_data()

# stratified sampling / split 
strat_training_set, strat_test_set = house_api.stratified_split_train_test(housing, split_ratio=0.2, random_seed=42)

# plot district map

# housing.plot( kind="scatter", x='longitude', y='latitude', colorbar=True,  colormap=colormaps["jet"] , c="median_house_value" ,marker='o', s= housing["population"]/100 , alpha = 0.3 ) # s = population size, c = color mapping to field 
# housing.hist( column= housing.keys().to_list()[ 2: :] , bins=50, figsize=(20, 15))
# plt.show()

# copy training set 

housing = strat_training_set.drop( columns=["median_house_value"] ).copy()
housing_labels = strat_training_set["median_house_value"].copy()
housing_num = housing.drop( columns=["ocean_proximity"], axis=1 )

# Pipeline sequence of transformation

num_pipeline =  Pipeline( [ ('imputer', SimpleImputer(strategy="median")), ("attribs_adder", house_api.CombinedAttributesAdder() ), ("std_scaler", std_scaler) ] )
num_attribs = list(housing_num) #  ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs), ("cat", one_hot_encoder, cat_attribs)]) 
housing_prepared = full_pipeline.fit_transform(housing)

# train and eval training set ( create model )

#   generate model (i.e. fit)
lin_reg.fit(housing_prepared, housing_labels)

#   quick samples to test training data
some_training_data = housing.iloc[:5] 
some_training_labels = housing_labels.iloc[:5]

#   transform training data (i.e. standardization, fill void or spare data sections, etc)
some_training_data_prepared = full_pipeline.transform(some_training_data)

#   predict using  model
some_training_predictions = lin_reg.predict(some_training_data_prepared)

#    root mean square error 
lin_reg_mse = mean_squared_error(some_training_labels, some_training_predictions)
lin_reg_mse = np.sqrt(lin_reg_mse)

print("Predictions: {}\t Labels: {}\t mse: {}".format(some_training_predictions, list(some_training_labels), lin_reg_mse))




