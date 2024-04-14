#!/usr/bin/env python 

"""
    Learn loading and preprocessing 

    Usage: notes.py 
"""

import os 
import tensorflow as tf
import matplotlib.pyplot as plt 

# basic data preprocessing  

tf.random.set_seed(seed=42)
rnd = tf.random.normal(shape=(1,))

# create dataset 
data_api = tf.data

dataset = data_api.Dataset.range(10)# random(seed=42).take(count=10)

# functional 
dataset = dataset.repeat(3).batch(7, drop_remainder=True)

# square batch data each packet
dataset = dataset.map(lambda x: x ** 2)

# unbatch this batched object 
dataset = dataset.apply(tf.data.experimental.unbatch())

# filter 
dataset = dataset.filter(lambda element: element > 10)

# shuffle
shuffle_buffer_size = 10
dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed =42).batch(batch_size=7)

# for item in dataset:
    # print(item)



