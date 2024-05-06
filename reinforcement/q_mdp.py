"""Using mdp run Q-Value iteration algorithm"""

import numpy as np 
import os
import sys 
from mdp import possible_actions, rewards, transition_probabilities

NUMBER_OF_STATES = 3
MAX_N_OF_ACTIONS_IN_STATE = 3
GAMMA = 0.90 # discount factor, higher implies agent values future rewards 
ITERATIONS = 50

Q_values = np.full(shape=(NUMBER_OF_STATES, MAX_N_OF_ACTIONS_IN_STATE), fill_value= -np.inf) 
for s, a in enumerate(possible_actions): 
    # s - state , a - action 
    Q_values[s, a] = 0.0

for i in range(ITERATIONS):
    Q_prev = Q_values.copy()
    for s in range(NUMBER_OF_STATES):
        for a in possible_actions[s]:
            Q_values[s, a] = np.sum([
                transition_probabilities[s][a][s_next] * (rewards[s][a][s_next] + GAMMA * np.max(Q_prev[s_next]) ) for s_next in range(MAX_N_OF_ACTIONS_IN_STATE)
            ])
print(Q_values)
