import os
import tarfile
import urllib
import urllib.request
import pandas as pd
import numpy as np 
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz" # https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz 

rooms_idx, bedrooms_idx, population_idx, households_idx = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y = None):
        return self 
    
    def transform(self, X):
        rooms_per_household = X[:, rooms_idx] / X[:, households_idx]
        population_per_household = X[:, population_idx] / X[:, households_idx]
        bedrooms_per_room = X[:, bedrooms_idx] / X[:, rooms_idx]
        new_attr_data = np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]        
        if not self.add_bedrooms_per_room:
            new_attr_data = new_attr_data[:, range(0, new_attr_data.shape[1] - 1) ] # numpy indexing + slice
        return new_attr_data

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path) # file located at url is archived to local disk   
    housing_tgz = tarfile.open(tgz_path) # open archive file 
    housing_tgz.extractall(  path=housing_path ) 
    housing_tgz.close()  # close archive file 

def load_housing_data(housing_path = HOUSING_PATH):
    sym_link_path_house_csv = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../Data/housing.csv' ) 
    local_path_house_csv =  os.path.join(housing_path, "housing.csv")

    csv_path = local_path_house_csv

    if os.path.islink(sym_link_path_house_csv):
        csv_path = sym_link_path_house_csv
    elif not os.path.exists(local_path_house_csv):
        fetch_housing_data()
        
    return pd.read_csv(csv_path) 

def split_train_test(data: pd.DataFrame, test_ratio: float):
    # this would break if dataset was refreshed 
    rng = np.random.default_rng(seed=42)
    shuffled_indices =  rng.permutation(len(data))
    test_set_size = int(len(data) * test_ratio) 
    test_indices = shuffled_indices[ : test_set_size] 
    train_indices = shuffled_indices[ test_set_size : ]
    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check(id, ratio):
    # computes hash of each id, returms true if has(id) is lower than 20% of maximum hash value (2**32)
    return crc32( np.int64(id)) & 0xFFFFFFFF < ratio * 2**32 

def split_train_test_by_id(data, ratio, column_attribute):
    # new training data instances must be appended to old datasheet 
    ids = data[column_attribute]
    in_test_set = ids.apply(lambda id: test_set_check(id, ratio))
    return data.loc[~in_test_set], data.loc[in_test_set] # training_set , test_set 

def stratified_split_train_test(data: pd.DataFrame, split_ratio: np.float32, random_seed: np.int32 = 42) -> tuple:
    data["income_cat"] = pd.cut(data["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=range(1, 6))
    split = StratifiedShuffleSplit(n_splits=1, test_size=split_ratio, random_state=random_seed)

    for train_indices, test_indices in split.split(data, data["income_cat"]) :
        strat_train_set = data.loc[train_indices]
        strat_test_set = data.loc[test_indices]
    
    # remove "income_cat" field from both sets
    for data_set_ in (strat_test_set, strat_train_set):
        data_set_.drop('income_cat', axis=1, inplace=True)

    return strat_train_set, strat_train_set

