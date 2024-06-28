# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:17:30 2024

@author: Xiaotong

test the algorithms in paper "Finite-time Analysis of the Multiarmed Bandit Problem"
"""

from masterTester import test
import sys
sys.path.append('../')

from environment.Gaussian_noise import Gaussian_noise as gn
from environment.env import env_stochastic
from environment.envAd1 import env_adverse1
from environment.envAd2 import env_adverse2
from environment.envAd3 import env_adverse3
from algorithms.exp3 import exp3
from algorithms.epsilion import EpsilonGreedy
from algorithms.UCB1 import UCB1

import numpy as np

num_arm = 5
T = 1000

# Stochastic
mu = np.array([0.1, 0.9, 0.4, 0.3, 0.6])
Trial = 10
noise = gn(1,0,0.01,[-0.1,0.1])
envs = list()
envs.append(env_stochastic(num_arm, mu, noise))
envs.append(env_adverse1(num_arm, noise))
envs.append(env_adverse2(num_arm, noise))
envs.append(env_adverse3(num_arm, noise, difficulty=100))


algorithms = [exp3(T, num_arm, alpha=0.1, gamma=0.3), UCB1(T, num_arm)]
algorithm_names = ['exp3', 'UCB1']
env_names = ['stochastic', 'biggerBetter', 'random', 'shifting']

test(T, Trial, envs, algorithms, algorithm_names, env_names)
