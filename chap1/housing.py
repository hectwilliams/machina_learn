import sys 
import os
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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, truncnorm 
import pickle

lin_reg = LinearRegression()
tree_reg = DecisionTreeRegressor()
forest_reg = RandomForestRegressor() #time consuming om my system !
std_scaler = StandardScaler()
ord_encoder = OrdinalEncoder()
one_hot_encoder = OneHotEncoder()
housing = house_api.load_housing_data()

# stratified sampling / split 
strat_training_set, strat_test_set = house_api.stratified_split_train_test(housing, split_ratio=0.2, random_seed=42)

# copy training set 

housing = strat_training_set.drop( columns=["median_house_value"] ).copy()
housing_labels = strat_training_set["median_house_value"].copy()
housing_num = housing.drop( columns=["ocean_proximity"], axis=1 )

#copy test set 

housing_test = strat_test_set.drop(columns=["median_house_value"]).copy()
housing_test_labels = strat_training_set["median_house_value"].copy()

# Pipeline sequence of transformation

num_pipeline =  Pipeline( [ ('imputer', SimpleImputer(strategy="median")), ("attribs_adder", house_api.CombinedAttributesAdder() ), ("std_scaler", std_scaler) ] )
num_attribs = list(housing_num) #  ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs), ("cat", one_hot_encoder, cat_attribs)]) 
housing_prepared = full_pipeline.fit_transform(housing)
housing_test_prepared = full_pipeline.fit_transform(housing_test)

# train and eval training set ( create model )

some_training_data = housing.iloc[:5] 
some_training_labels = housing_labels.iloc[:5]

#   transform training data (i.e. standardization, fill void or spare data sections, etc)
some_training_data_prepared = full_pipeline.transform(some_training_data)

models = {
    "forest_reg": {
        'cv': 5,
        'random_state': 0,
        'filename': 'random_forest',
        'estimator': RandomForestRegressor,
        'search_function': RandomizedSearchCV,
        'scoring': 'neg_mean_squared_error',
        'param': {
            'n_estimators': [3, 10, 30],
            'max_features': [2,3,4],
            'max_depth': [5,10],
            'min_samples_split': [2,5,10],
            'min_samples_leaf': [1,2,4],
            'bootstrap': [True, False]
        },
        'iter':5,
        'verbose':0
    },
}

model_stuct = models['forest_reg']

# random search

hyper_param_search_cv = model_stuct['search_function']( verbose=model_stuct['verbose'], n_iter = model_stuct['iter'], estimator=model_stuct['estimator']() , param_distributions=model_stuct['param'], cv =model_stuct['cv'], scoring=model_stuct['scoring'], return_train_score=True, random_state=model_stuct['random_state'])
hyper_param_search_cv.fit(housing_prepared, housing_labels)

# inspect model

cv_results = hyper_param_search_cv.cv_results_

model = hyper_param_search_cv.best_estimator_

best_param = hyper_param_search_cv.best_params_

feature_important = model.feature_importances_

# use model (i.e. deployed)

final_predictions = model.predict(housing_test_prepared)
final_mse = mean_squared_error(housing_test_labels, final_predictions)
final_mse = np.sqrt(final_mse)

# attributes
extra_attribs = ['rooms_per_household', 'population_per_household', 'bedrooms_per_household']
cat_one_hot_attribs = list((full_pipeline.named_transformers_['cat']).categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs

# rank attributes 
attrib = list(zip(feature_important, attributes))
attrib.sort(key = lambda tupe: tupe[0] , reverse=True)

print("mse={}, best_param=\n{}\n\nfeature_important\n{}\n".format(final_mse, best_param, attrib))

# save model 

file_output = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_stuct['filename']  ) 

with open( "{}.pkl".format(file_output), "wb") as file_handle:
    pickle.dump(  { 'attributes': attrib, 'hyper_param': hyper_param_search_cv, "predictions": final_predictions , "mse": final_mse , "model": model}, file_handle)

