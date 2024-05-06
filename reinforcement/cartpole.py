"""Using OpenAI Gym and create a CartPole environment 
"""
import gym 
import matplotlib.pyplot as plt
import numpy as np 
import sys 

GYM_ID = "CartPole-v1"
NUM_POLICY_RUNS = 5
NUM_ACTIONS_PER_POLICY = 50

plt.ion()
plt.show()

def plot_image(img):
    """Plots latest render of Cartpole
    
    
    Args:
        img: Image to plot
    """
    plt.imshow(img)
    plt.draw()
    plt.pause(0.01)

def cart_policy(obs):
    """Controls direction 'action' taken by cart


    Args:
        obs: vector containing the following:
            -0- horizontal position
            -1- velocity
            -2- pole angle
            -3- angular velocity
    
            
    Returns:
        Binary action value: 
            0 - accelerate left
            1 - accelerate right
    """
    return 1 if obs[2] > 0 else 0

env = gym.make(id=GYM_ID, render_mode="rgb_array")
rewards_list = []

for run_iteration in range(NUM_POLICY_RUNS):
    curr_rewards = 0
    obs, info = env.reset() # [horizontal position, velocity position, angle of pole, angular velocity]
    for run_action in range(NUM_ACTIONS_PER_POLICY):
        print(f'run {run_action}')
        plot_image(env.render())
        action = cart_policy(obs)
        obs, reward, done, trunc, info  = env.step(action=action) 
        curr_rewards += reward
        if done:
            break 
    rewards_list.append(curr_rewards)
    print(f'next policy experiment {run_iteration}')
rewards_numpy = np.array(rewards_list)
mean = np.mean(rewards_numpy)
std = np.std(rewards_numpy)
min = np.min(rewards_numpy)
max = np.max(rewards_numpy)

print(f'mean:{mean}\nstd:{std}\nmin{min}\nmax{max}')

plt.ioff()
plt.show()