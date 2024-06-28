from masterTester import test
import sys
sys.path.append('../')

from environment.Gaussian_noise import Gaussian_noise as gn
from environment.env import env_stochastic
from algorithms.exp3M import exp3M

import numpy as np

num_arm = 10
batch_size = 3
T = 1000

# Stochastic
mu = np.array([0.1, 0.9, 0.4, 0.3, 0.6, 0.1, 0.001, 0.95, 0.7, 0.6])
Trial = 10
noise = gn(1,0,0.01,[-0.1,0.1])
envs = list()
envs.append(env_stochastic(num_arm, mu, noise))


algorithms = [exp3M(T, num_arm, batch_size, gamma=0.1)]
algorithm_names = ['exp3M']
env_names = ['stochastic', 'biggerBetter', 'random', 'shifting']

test(T, Trial, envs, algorithms, algorithm_names, env_names)
