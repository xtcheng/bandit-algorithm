# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:42:42 2024

@author: Xiaotong

test the algorithms in paper "Distributed cooperative decision making in multi-agent multi-armed bandits"
"""

# main.py
import numpy as np
import matplotlib.pyplot as plt

from environment.Gaussian_noise import Gaussian_noise as gn
from environment.env import env_stochastic

from algorithms.CoopUCB2sl import CoopUCB2Selective
from algorithms.CoopUCB2 import CoopUCB2

# Parameters
T = 5000  # Number of trials
num_agents = 5
num_arms = 10
mu = [0.4, 0.5, 0.5, 0.6, 0.7, 0.7, 0.8, 0.9, 0.92, 0.95]
observation_matrix = [
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0 ],
]
gamma = 1
eta = 0.5
sigma_g = 2.5
# Initialize environment and algorithm

noise = gn(1,0,0.01,[-0.1,0.1])
env = env_stochastic(num_arms, mu, noise)

algorithm = CoopUCB2(T, num_agents, num_arms, observation_matrix, gamma, sigma_g, eta)
algorithm2 = CoopUCB2Selective(T, num_agents, num_arms, observation_matrix, gamma, sigma_g, eta)

# Run the algorithm
algorithm.run(env)
algorithm2.run(env)
# Get average regret
avg_regret = algorithm.get_avg_rgt()
cum_regret = algorithm.get_cum_rgt()

avg_regret2 = algorithm2.get_avg_rgt()
cum_regret2 = algorithm2.get_cum_rgt()


# Plotting
plt.plot(avg_regret)
plt.xlabel('Trials')
plt.ylabel('Average Regret')
plt.title('Performance of coop_ucb Algorithm')
plt.grid(True)
plt.show()

# Plotting
plt.plot(cum_regret)
plt.xlabel('Trials')
plt.ylabel('Cummulative Regret')
plt.title('Performance of coop_ucb Algorithm')
plt.grid(True)
plt.show()

fig, axs = plt.subplots( 2,figsize=(12, 5))
# Plot cumulative regret for each agent for Case 1
for agent in range(num_agents):
    axs[0].plot(algorithm.cumulative_regret[agent], label=f'Agent {agent+1}')
axs[0].set_title('Cumulative Regret of coopUCB Algorithm')
axs[0].set_xlabel('Trials')
axs[0].set_ylabel('Cumulative Regret')
axs[0].legend()
axs[0].grid(True)

for agent in range(num_agents):
    axs[1].plot(algorithm.average_regret[agent], label=f'Agent {agent+1}')
axs[1].set_title('Cumulative Regret of coopUCB Algorithm')
axs[1].set_xlabel('Trials')
axs[1].set_ylabel('Average Regret')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

# Plotting
plt.plot(avg_regret2)
plt.xlabel('Trials')
plt.ylabel('Average Regret')
plt.title('Performance of coop_ucb Algorithm')
plt.grid(True)
plt.show()

# Plotting
plt.plot(cum_regret2)
plt.xlabel('Trials')
plt.ylabel('Cummulative Regret')
plt.title('Performance of coop_ucb Algorithm')
plt.grid(True)
plt.show()

fig, axs = plt.subplots( 2,figsize=(12, 5))
# Plot cumulative regret for each agent for Case 1
for agent in range(num_agents):
    axs[0].plot(algorithm2.cumulative_regret[agent], label=f'Agent {agent+1}')
axs[0].set_title('Cumulative Regret of coopUCB Algorithm')
axs[0].set_xlabel('Trials')
axs[0].set_ylabel('Cumulative Regret')
axs[0].legend()
axs[0].grid(True)

for agent in range(num_agents):
    axs[1].plot(algorithm2.average_regret[agent], label=f'Agent {agent+1}')
axs[1].set_title('Average Regret of coopUCB Algorithm')
axs[1].set_xlabel('Trials')
axs[1].set_ylabel('Average Regret')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()