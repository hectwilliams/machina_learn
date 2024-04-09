#!/usr/bin/env python 

"""
    Example of Autodiff to compute gradients 
  

    Usage: Compute_2D_cost.py 
"""

import os
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt  

def convolve_one_dimensional(x, y):
    size_n = tf.size(x)
    conv_length = size_n + (size_n-1)
    wave_convul =  tf.zeros(conv_length)
    
    for shift_iter in range(1,conv_length+1):
        if shift_iter <= size_n:
            x_active = x[:shift_iter]
            h_active = y[shift_iter - 1 :: -1]
        else:
            x_active = x[shift_iter - size_n :]
            h_active = y[:-(shift_iter - size_n)]

        x_h = tf.multiply(x_active, h_active)

        wave_convul = tf.tensor_scatter_nd_update( wave_convul, [ [shift_iter - 1] ] , [tf.reduce_sum(x_h)] ) 
        
    return  wave_convul

def cost(w1, w2):
    global wave_z
    global ts 
    global linespace_t 
    global N
    
    tone_a = tf.math.sin(2 * np.pi * w1 * (ts * linespace_t)) 
    tone_b = tf.math.sin(2 * np.pi * w2 * (ts * linespace_t))
    mixed_tone = tone_a + tone_b    #convolve_one_dimensional(tone_a, tone_b)

    return tf.divide( tf.reduce_sum(tf.square( mixed_tone - wave_z)),tf.cast(N, tf.float32))
    
rng = np.random.default_rng(seed=42)
curr_dir = os.path.join(os.path.dirname(os.path.realpath(__file__))) 

# tone para
freq  = tf.constant([5.0, 5.0], dtype=tf.float32)
fs = tf.constant(50,  dtype=tf.float32) # sample rate - 48 kHz 
ts = tf.constant( 1/fs , dtype=tf.float32) 
N = tf.cast(tf.constant(2 / ts), tf.int32)
linespace_t = tf.range(N, dtype=tf.float32) 
ts_samples = linespace_t * ts

# wave tomes (wave size = N )
wave_x = tf.math.sin(  2 * np.pi * freq[0] * ts_samples )
wave_y = tf.math.sin(  2 * np.pi * freq[1] * ts_samples )

# mix tones (convolve size = N + (N -1))
wave_z = wave_x + wave_y #convolve_one_dimensional(wave_x, wave_y) 

# guess wave
n = 100
f_linspace = tf.linspace(start = 0, stop = 10, num = n)
guess_x = tf.reshape(f_linspace, shape=(1,-1))
guess_x_t = tf.transpose(guess_x) 
guess_y = tf.reshape(f_linspace, shape=(-1, 1))
guess_z = np.zeros(shape=(n, n))

plt.figure()
plt.xlim(0, 10)
plt.ylim(0,10)
plt.xlabel('freq1')
plt.ylabel('freq2')
plt.title('Cost Function 2 variables')
for y_index in  range(n):
    for x_index in  range(n):
        w1 = tf.cast(guess_x_t[x_index], tf.float32)
        w2 = tf.cast(guess_y[y_index], tf.float32)
        data = tf.get_static_value(cost(w1, w2))
        guess_z = tf.tensor_scatter_nd_update(guess_z, [ [y_index, x_index] ], [data ])

# Plot cost function 
x1, x2 = np.meshgrid(f_linspace, f_linspace)
ln = plt.contourf( x1, x2, guess_z)
plt.colorbar()
plt.savefig(os.path.join(curr_dir, __file__[:-2]))


