"""Use Q-Learning algorithm to create agent that uses optimal policy to choose action with highest Q-Value
"""
import mdp
import numpy as np 
import matplotlib.pyplot as plt 

NUM_ITERATIONS = 10000
ALPHA0 = 0.05
DECAY = 0.005 
GAMMA = 0.90 

rng = np.random.default_rng(seed=42)
state = 0 # initial state 

def step(state, action):
    """Agent executes an action 
    
    
    Args:
        state: agent's state in markov decision process 
        action: action taken at current state
        
        
    Returns:
        Two element tuple containing next_state and reward, respectively
    """
    probas = mdp.transition_probabilities[state][action]
    next_state = rng.choice(3, p=probas)
    reward = mdp.rewards[state][action][next_state]
    return next_state, reward 

def exploration_policy(state):
    """Agent returns a random action
    
    
    Args:
        state: Agent's state in Markov Decision Process
        
        
    Returns:
        Random action
    """
    actions = mdp.possible_actions[state]
    return rng.choice(actions)

Q_values = np.full(shape=(3, 3), fill_value= -np.inf) 
data = Q_values.copy()[np.newaxis]

for s, a in enumerate(mdp.possible_actions): 
    Q_values[s, a] = 0.0

for iteration in range(NUM_ITERATIONS):
    action  = exploration_policy(state)
    next_state, reward = step(state, action) 
    next_Q_value = np.max(Q_values[next_state])
    # TD Learning Algorithm
    alpha  = ALPHA0 / (1 + iteration * DECAY)
    Q_values[state, action] *= 1 - alpha
    Q_values[state, action] += alpha * (reward + GAMMA * next_Q_value)
    if iteration == 0:
        data = Q_values.copy()[np.newaxis]
    else:
        data = np.vstack((Q_values[np.newaxis], data))
    state = next_state
a = data[:, 0, 0]
rev = a[::-1]
plt.plot(rev)
plt.ylabel( "Q value s{} a{}".format( '\u2080', '\u2080') )    
plt.xlabel(f'{"iteration"}')
plt.show()
