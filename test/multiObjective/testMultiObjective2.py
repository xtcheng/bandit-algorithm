import sys
sys.path.append('../../')

from helpers.masterTester import test

from environment.Gaussian_noise import Gaussian_noise as gn
from environment.multiOutput import EnvMultiOutput
from algorithms.modular.moduleUsers.expertsMultiObjective import ExpertsMultiObjective
from algorithms.modular.moduleUsers.fair_MO_UCB import Fair_MO_UCB

import numpy as np

if __name__ == "__main__":
	num_arm = 4
	num_objectives=5
	T = 10000
	
	
	mu = []
	mu.append(np.array([.15, .3, .5, .72, .8]))
	mu.append(np.array([.2, .18, .48, .7, .2]))
	mu.append(np.array([.15, .4, .3, .8, .4]))
	mu.append(np.array([.2, .2, .8, .3, .5]))
	
	weights = []
	for i in range(0, num_objectives):
		weights.append(1/(2**i))
	
	noise = gn(1,0,0.01,[-0.2,0.2])
	trial = 8
	envs = list()
	envs.append(EnvMultiOutput(num_arm, mu, noise, weights))

	algorithms = []
	algorithms.append(ExpertsMultiObjective(T, num_arm=num_arm, num_objectives=num_objectives, gini_weights=weights))
	algorithms.append(Fair_MO_UCB(T, num_arm=num_arm, num_objectives=num_objectives, delta=0.1, gini_weights=weights))
	
	algorithm_names = []
	algorithm_names.append("ExpertsMultiObjective")
	algorithm_names.append("Fair MO UCB")
	env_names = ["Normal Environment"]
	
	test(T, trial, envs, algorithms, algorithm_names, env_names)
