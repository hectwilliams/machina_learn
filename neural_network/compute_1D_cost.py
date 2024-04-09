#!/usr/bin/env python 

"""
    Example of Autodiff to compute gradients 

    Usage: Compute_1D_cost.py 
"""

import os
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt  

def cost(w1):
    global ts
    global linespace_t
    global y
    global N
    return tf.divide(tf.reduce_sum(tf.square(tf.math.sin(2 * np.pi * w1 * (ts * linespace_t)) - y)) ,tf.cast(N, tf.float32) )

rng = np.random.default_rng(seed=42)
wk_dir = os.path.join(os.path.dirname(os.path.realpath(__file__))) 

# EXAMPLE OF COST FUNCTION  finding frequency to match sin wave

# Simple sin wave 

f = 5
fs = tf.constant(48000,  dtype=tf.float32) # sample rate - 48 kHz 
ts = tf.constant( 1/fs , dtype=tf.float32) 
N = tf.cast(tf.constant(2 / ts), tf.int32)
linespace_t = tf.range(N, dtype=tf.float32) 
ts_samples = linespace_t * ts
y = tf.math.sin(  2 * np.pi * f * ts_samples ) 
n = 1000
f_linspace = np.linspace(start = 0, stop = 10, num = n, dtype=np.float32)

# Plot cost function 

cost_axis = np.zeros(n)
for i in range(n):
    cost_axis[i] = cost(w1=f_linspace[i])

plt.figure() 
ln = plt.plot(f_linspace, cost_axis, marker='.', alpha=0.01)
plt.xlabel("frequency_estimate")
plt.xlabel("frequency")
plt.ylabel('MSE Cost')
plt.title('Cost Function 1 Variable')
plt.xlim(0, 10)
plt.ion()
plt.show()

# Custom GD search for min cost 

f_guess = tf.Variable(1.2 )
lr = tf.constant(0.4)
eta = 0.9
buffer = []
cost_best_estimate = {'cost': 2**64, 'guess': 0, "bin_min" : 0, 'bin_max': 10}

max_guess = cost_best_estimate['bin_max'] 
min_guess = cost_best_estimate['bin_min'] 
delta = (max_guess - min_guess) / 10

for w in np.arange(start = min_guess, stop=max_guess, step=delta):
    low_limit = w 
    high_limit = w + delta

    for _ in range(10):
        guess = tf.Variable(rng.uniform(low_limit, high_limit))

        with tf.GradientTape(persistent=True) as tape:
            z_cost = cost(guess)
            gradient = tape.gradient(z_cost, guess)
            guess_next_step = guess - (lr * gradient)
            if guess_next_step >= low_limit and guess_next_step <= high_limit:
                guess = guess_next_step

            if i == 0:
                plt.scatter(guess, z_cost, marker='.', s=1, color='red')
            else:
                artist_next = plt.scatter(guess, z_cost, marker='.', s=1, c="orange")
            
            if z_cost < cost_best_estimate['cost'] :
                cost_best_estimate['cost'] = z_cost
                cost_best_estimate['guess'] = guess
                cost_best_estimate['bin_min'] = low_limit
                cost_best_estimate['bin_max'] = high_limit

            plt.draw()
            plt.pause(0.2)

plt.scatter(cost_best_estimate['guess'], cost_best_estimate['cost'], c='blue' , marker='x', s=10)

plt.draw()
plt.pause(0.2)

plt.ioff()
plt.savefig(os.path.join(wk_dir, __file__[:-2]))



