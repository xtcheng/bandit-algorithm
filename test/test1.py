# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:17:30 2024

@author: Xiaotong

test the algorithms in paper "Finite-time Analysis of the Multiarmed Bandit Problem"
"""

import sys
sys.path.append('../')

from environment.Gaussian_noise import Gaussian_noise as gn
from environment.env import env_stochastic
from algorithms.UCB2 import UCB2
from algorithms.epsilion import EpsilonGreedy
from algorithms.UCB1 import UCB1
from algorithms.UCB1N import UCB1N 

import numpy as np
import matplotlib.pyplot as plt

num_arm = 5
T = 100

mu = np.array([0.1, 0.9, 0.4, 0.3, 0.6])
Trial = 10
noise = gn(1,0,0.01,[-0.1,0.1])
env = env_stochastic(num_arm, mu, noise)


algorithms = [UCB2(T, num_arm, alpha=0.1), EpsilonGreedy(T, num_arm, c=0.01, d=0.05), UCB1(T, num_arm), UCB1N(T, num_arm)]
algorithm_names = ['UCB2', 'Epsilon-Greedy', 'UCB1', 'UCB1N']
avg_regret = []

for i, algorithm in enumerate(algorithms):
    regret_sum = 0
    for trial in range(Trial):
        algorithm.clear()
        algorithm.run(env)
        regret_sum += algorithm.get_cum_rgt()[-1]
    avg_regret.append(regret_sum / Trial)

plt.figure(figsize=(6, 5))
for i, algorithm in enumerate(algorithms):
    plt.plot(range(T), algorithm.get_cum_rgt(), label=algorithm_names[i])
plt.xlabel('t (Trials)', fontsize=15)
plt.ylabel('Cumulative Regret', fontsize=15)
plt.legend(loc='upper right')
plt.title('Cumulative Regret')
#plt.show()


plt.figure(figsize=(6, 5))
for i, algorithm in enumerate(algorithms):
    plt.plot(range(T), algorithm.get_avg_rgt(), label=algorithm_names[i])
plt.xlabel('t (Trials)', fontsize=15)
plt.ylabel('Average Regret', fontsize=15)
plt.legend(loc='upper right')
plt.title('Average Regret')
plt.show()
