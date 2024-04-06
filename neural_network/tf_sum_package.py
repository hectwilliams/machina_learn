#!/usr/bin/env python 

"""
    Use SummaryWriter to log data

    Usage: tf_sum_package.py
"""

import os
import sys 
import time 
import keras 
import numpy as np 
import tensorflow as tf 
from matplotlib.image import imread

def get_tensor_log_id(cur_dir):
    '''
        tf logs
    '''
    date_id = time.strftime("run_%Y_%m_%d__%H_%M_%S") # time to string
    return os.path.join(cur_dir, "sum_pkg_logs", date_id)

rng = np.random.default_rng(seed=42)
current_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)) ) 
test_logdir = get_tensor_log_id(current_dir)
writer = tf.summary.create_file_writer(logdir=test_logdir)

with writer.as_default():
    for step in range(1, 5):

        # sin wave 
        tf.summary.scalar( name= "my_scalar", data= np.sin(step/10), step= step)

        # histo 
        data = (rng.standard_normal(size=(1, 100))) * step 
        tf.summary.histogram("my_hist",data,buckets=50, step=step)

        # image 
        img = imread(os.path.join(current_dir, "..","unsupervised_learn", "images", "ladybug.png")) # img.shape (1452, 1452, 3) row, col, rgb
        img_tf = tf.convert_to_tensor( [img], np.float32)
        tf.summary.image("my_images", img_tf, step=1)

        # texts 
        texts = [ f'The step in {str(step)}' ]
        tf.summary.text("my_test", texts, step=step)
        
        # audio wave
        N = 10560000 
        fs = tf.constant(44.1e3, dtype=tf.float32) 
        ts = tf.constant(1/fs)
        ts_samples = tf.range(N, dtype=tf.float32)  * ts
        sin_wave = tf.math.sin(2*np.pi*step*ts_samples) * rng.integers(low=1, high=10)
        audio = tf.reshape((sin_wave), [1,-1,1])
        audio = tf.cast(audio, dtype=tf.float32)
        tf.summary.audio(name= "my_audio" , data= audio, sample_rate=tf.constant( np.int32(fs) ) , step=step)
      
        



