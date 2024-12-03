# -*- coding: utf-8 -*-
"""
Created on Fri May 31 12:00:06 2024

test algorithm in "Combinatorial Network Optimization With Unknown Variables: Multi-Armed BanditsWith Linear Rewards and Individual Observations"

@author: Xiaotong
"""

import numpy as np
import matplotlib.pyplot as plt

from algorithms.mucb import MUCB
from environment.Gaussian_noise import Gaussian_noise as gn
from environment.env import env_stochastic

num_agents = 4
num_arms = 6
num_trials = 1000
ucb1 = MUCB(num_agents, num_trials, num_arms)
mu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8]  

noise = gn(1,0,0.01,[-0.1,0.1])
env = env_stochastic(num_arms, mu, noise)

num_simulations = 1

avg_cumulative_regret = np.zeros(num_trials)
avg_average_regret = np.zeros(num_trials)

for _ in range(num_simulations):
    ucb1 = MUCB(num_agents, num_trials, num_arms)
    ucb1.run(env)
    avg_cumulative_regret += ucb1.get_cum_rgt()
    avg_average_regret += ucb1.get_avg_rgt()
    ucb1.clear()


avg_cumulative_regret /= num_simulations
avg_average_regret /= num_simulations

plt.figure(figsize=(10, 5))
plt.plot(range(num_trials), avg_average_regret)
plt.title('Average Regret Over Time')
plt.xlabel('Time Step')
plt.ylabel('Average Regret')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(num_trials), avg_cumulative_regret)
plt.title('Cumulative Regret Over Time')
plt.xlabel('Time Step')
plt.ylabel('Cumulative Regret')
plt.show()
