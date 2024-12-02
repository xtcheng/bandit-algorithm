import sys
sys.path.append('../')

from helpers.masterTester import test

from environment.Gaussian_noise import Gaussian_noise as gn
from environment.env import env_stochastic
from algorithms.UCB1 import UCB1
from algorithms.modular.moduleUsers.ucb import ModularUCB

import numpy as np

if __name__ == "__main__":
	num_arm = 5
	T = 1000

	# Stochastic
	mu = np.array([0.1, 0.9, 0.4, 0.3, 0.6])
	Trial = 10
	noise = gn(1,0,0.01,[-0.1,0.1])
	envs = list()
	envs.append(env_stochastic(num_arm, mu, noise))


	algorithms = [UCB1(T, num_arm,2), ModularUCB(T, num_arm,2)]
	algorithm_names = ['Original UCB', 'Modular implementation']
	env_names = ['stochastic']

	test(T, Trial, envs, algorithms, algorithm_names, env_names)
