#!/usr/bin/env python 

"""
    Embedding example using housing dataset 

    Usage: embed.py 
"""

import os 
import sys 
import numpy as np
import tensorflow as tf 
import keras 

wk_dir = os.path.join(os.path.dirname(os.path.realpath(__file__))) 
sys.path.append( os.path.join(wk_dir,"../learning_project_end_to_end/") )

import housing_download as house_api

rng = np.random.default_rng(seed=42)
housing_dataset = house_api.load_housing_data()
validation_percent = 0.10
validation_size = np.int32(len(housing_dataset) * validation_percent)

# Get feature(s)

# training set variables 
strat_training_set, strat_test_set = house_api.stratified_split_train_test(housing_dataset, split_ratio=0.2, random_seed=42)
housing = strat_training_set.drop( columns=['median_house_value', 'ocean_proximity'] ).copy()
housing_labels = strat_training_set['median_house_value'].copy()
housing_ocean_proximity = strat_training_set['ocean_proximity'].copy()

# test set variables
housing_test = strat_test_set.drop( columns=['median_house_value', 'ocean_proximity'] ).copy()
housing_labels_test = strat_test_set['median_house_value'].copy()
housing_ocean_proximity_test = strat_test_set['ocean_proximity'].copy()

# validation data 
housing_validation, housing_train = housing[:validation_size], housing[validation_size:]
housing_labels_validation, housing_labels_train = housing_labels[:validation_size], housing_labels[validation_size:]
housing_ocean_proximity_validation, housing_ocean_proximity_train = housing_ocean_proximity[:validation_size] , housing_ocean_proximity[validation_size:]

# Map categories to indices 
feature_names = list(housing)
category_names = strat_training_set['ocean_proximity'].unique()
category_names_indices = tf.range(len(category_names), dtype=tf.int64)
table_init = tf.lookup.KeyValueTensorInitializer(category_names, category_names_indices)
num_oov_buckets = 2
table = tf.lookup.StaticVocabularyTable(initializer=table_init, num_oov_buckets=num_oov_buckets)

# Build Model

    # Standardize Layer

regular_inputs = keras.layers.Input(shape = housing_train.shape[1:], name="standardized_data" ) # 8 channels

length_percent_40_train = int(0.4 * len(housing_train)) 
rand_indices = rng.integers(low =1, high=len(housing_train), size=(length_percent_40_train))
adapt_values = housing_train.iloc[rand_indices]

standardization = keras.layers.Normalization()
standardization.adapt(data = adapt_values.to_numpy())
standardized_input = standardization(regular_inputs) 

    # Embedding Layer 

embedding_dimension = 2
categories = keras.layers.Input(shape=housing_ocean_proximity_train.shape[1:], dtype=tf.string, name= "embeddings") # 1 channels
category_to_indices = keras.layers.Lambda( function = lambda cat: table.lookup( cat  ) , output_shape=[] ) (categories)
indices_to_embeddings = keras.layers.Embedding( input_dim= len(category_names) + num_oov_buckets , output_dim=embedding_dimension)  (category_to_indices)

    # Link Layers

encoded_category = keras.layers.concatenate([standardized_input, indices_to_embeddings])
outputs = keras.layers.Dense(1)(encoded_category)
model = keras.models.Model(inputs=[regular_inputs, categories], outputs=[outputs])

    # Build Model

model.compile(loss=keras.losses.mean_squared_error, optimizer= keras.optimizers.SGD())

model_history = model.fit( x = [housing_train, housing_ocean_proximity_train] , y= [housing_labels_train])

    # Plot Model 

keras.utils.plot_model(model, os.path.join(wk_dir, __file__ + '.png'), show_shapes=True)
