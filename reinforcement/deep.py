import sys 
import keras 
import numpy as np
import gym 
from collections import deque 
import tensorflow as tf 
import matplotlib.pyplot as plt
import os 

GYM_ID = "CartPole-v0"
INPUT_SIZE = 4 # observation parameters
N_OUTPUTS = 2
ACTIVATION = "elu"
BATCH_SIZE = 32
DISCOUNT_FACTOR = 0.95
NUM_EPISODES = 600
NUM_OF_STEPS = 200
DEQUE_MAX_LEN = 5000
CURR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__))) 

env = gym.make("CartPole-v0")
rng = np.random.default_rng(seed=42)
buffer = deque(maxlen=DEQUE_MAX_LEN)
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = keras.losses.mean_squared_error
model = keras.Sequential(
    [
        keras.layers.Dense(32, activation=ACTIVATION, input_shape=[INPUT_SIZE]),
        keras.layers.Dense(32, activation=ACTIVATION),
        keras.layers.Dense(N_OUTPUTS),
    ]
)  
model.compile(loss=loss_fn, optimizer=optimizer)
target_model = keras.models.clone_model(model)
target_model.set_weights(model.get_weights())

plt.ion()
plt.show()
plt.xlim(0, NUM_EPISODES)
plt.ylim(0, NUM_OF_STEPS)
plt.xlabel("EPISODE")
plt.ylabel("SUM OF REQARDS")

def epsilon_greedy_policy(state, epsilon=0):
    """ Pick action with largest predicted Q value
    
    
    Args:
        state: Agent's state
        epsilon: Probability
    
        
    Returns:
        Agent's action 
    """
    if rng.random() < epsilon:
        return rng.integers(2)
    else:
        Q_values = model.predict( state[np.newaxis] )
        return np.argmax(Q_values)

def sample_experiences(batch_size):
    """ Sample random batch of experiences
    
    
    Args:
        batch_size = Integer value 
    
    
    Returns:
        Five NumPy arrays containing five experience elements
    """
    indices = rng.integers(low=0, high =len(buffer), size=batch_size)
    batch = [ buffer[i]  for i in indices ]
    states, actions, rewards, next_states, dones = [
        np.array( [experience[i] for experience in batch] ) for i in range(5)
    ]
    return states, actions, rewards, next_states, dones

def play_one_step(env, state, epsilon):
    """ Play single step using epsilon
    
    
    Args:
        env:  CartPole environment 
        state: Observation  
        epsilon: Probability
    
    
    Return:
        Array containing next_state, reward, done, info
    """
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

def training_step(batch_size):
    """Performs Gradient Descent Step on batch
    
    
    Args:
        batch_size: Integer value
    
    """
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    # predict next_states Q_values
    next_Q_values = target_model.predict(next_states)
    max_next_Q_values = np.max(next_Q_values, axis = 1)
    target_Q_values = rewards + (1-dones) * DISCOUNT_FACTOR * max_next_Q_values
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, N_OUTPUTS)

    with tf.GradientTape() as tape:
        Q_values_ = model(states)
        Q_values = tf.reduce_sum(Q_values_ * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))    
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
for episode in range(NUM_EPISODES):
    r = 0
    obs = env.reset()
    for step in range(NUM_OF_STEPS):
        epsilon = max ( 1 - episode / 500, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        r += reward
        if done:
            break 
    
    plt.scatter(episode, r, marker='.', s=1)
    plt.draw() 
    plt.pause(0.2) 
    
    if episode > 50:
        training_step(BATCH_SIZE)
    
    if episode % 50 == 0:
        target_model.set_weights(model.get_weights())
    

plt.savefig(  os.path.join(CURR_DIR, __file__ [:-3] + "_learning_curve_" + '.png')   )
plt.ioff()
