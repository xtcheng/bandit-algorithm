# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:51:09 2024
test algorithm in "Heterogeneous Stochastic Interactions for Multiple Agents in a Multi-armed Bandit Problem"
@author: Xiaotong
"""

# main.py
import numpy as np
import matplotlib.pyplot as plt
from environment.Gaussian_noise import Gaussian_noise as gn
from environment.env import env_stochastic

from algorithms.MAMAB_SI import MAMAB_SI

# Parameters
T = 50000  # Number of trials
num_agents = 6
num_arms = 10
mu = [0.4, 0.5, 0.5, 0.6, 0.7, 0.7, 0.8, 0.9, 0.92, 0.95]
sociabilities = [0.50, 0.85, 0.05, 0.50, 1.00, 0.90]
observation_matrix = [
    [0, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0]
]

# Initialize environment and algorithm
noise = gn(1,0,0.01,[-0.1,0.1])
env = env_stochastic(num_arms, mu, noise)
algorithm = MAMAB_SI(T, num_agents, num_arms, sociabilities, observation_matrix)

# Run the algorithm
algorithm.run(env)

# Get average regret
avg_regret = algorithm.get_avg_rgt()
cum_regret = algorithm.get_cum_rgt()


# Plotting
plt.plot(avg_regret)
plt.xlabel('Trials')
plt.ylabel('Average Regret')
plt.title('Performance of MAMAB_SI Algorithm')
plt.grid(True)
plt.show()

# Plotting
plt.plot(cum_regret)
plt.xlabel('Trials')
plt.ylabel('Average Regret')
plt.title('Performance of MAMAB_SI Algorithm')
plt.grid(True)
plt.show()
