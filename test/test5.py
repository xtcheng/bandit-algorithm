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
from environment.envAd1 import env_adverse1
from algorithms.exp3 import exp3
from algorithms.epsilion import EpsilonGreedy
from algorithms.UCB1 import UCB1

import numpy as np
import matplotlib.pyplot as plt

num_arm = 5
T = 100

# Stochastic
mu = np.array([0.1, 0.9, 0.4, 0.3, 0.6])
Trial = 10
noise = gn(1,0,0.01,[-0.1,0.1])
envs = list()
envs.append(env_stochastic(num_arm, mu, noise))


algorithms = [exp3(T, num_arm, alpha=0.1, gamma=0.8), UCB1(T, num_arm)]
algorithm_names = ['exp3', 'UCB1']
env_names = ['stochastic']
avg_regret = []
cum_regret = []

for env in envs:
    for i, algorithm in enumerate(algorithms):
        cum_regret.append([0]*T)
        avg_regret.append([0]*T)
        for trial in range(Trial):
            algorithm.clear()
            algorithm.run(env)
            for y in range(T):
                cum_regret[-1][y] += algorithm.get_cum_rgt()[y] / Trial
                avg_regret[-1][y] += algorithm.get_avg_rgt()[y] / Trial

plt.figure(figsize=(6, 5))
for j, env in enumerate(envs):
    for i, algorithm in enumerate(algorithms):
        plt.plot(range(T), cum_regret[len(algorithms)*j + i], label=algorithm_names[i]+" on "+env_names[j])
plt.xlabel('t (Trials)', fontsize=15)
plt.ylabel('Cumulative Regret', fontsize=15)
plt.legend(loc='upper right')
plt.title('Cumulative Regret')
#plt.show()


plt.figure(figsize=(6, 5))
for j, env in enumerate(envs):
    for i, algorithm in enumerate(algorithms):
        plt.plot(range(T), avg_regret[len(algorithms)*j + i], label=algorithm_names[i]+" on "+env_names[j])
plt.xlabel('t (Trials)', fontsize=15)
plt.ylabel('Average Regret', fontsize=15)
plt.legend(loc='upper right')
plt.title('Average Regret')
plt.show()
