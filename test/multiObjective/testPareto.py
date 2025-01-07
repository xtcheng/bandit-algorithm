import sys
sys.path.append('../../')

from helpers.masterTester import test

from environment.Gaussian_noise import Gaussian_noise as gn
from environment.multiOutput import EnvMultiOutput
from algorithms.modular.moduleUsers.paretoUCB import ParetoUCB
from algorithms.modular.moduleUsers.basicMultiObjective import BasicMultiObjective

import numpy as np

if __name__ == "__main__":
	num_arm = 4
	T = 3000

	weights = [1, 1/2]

	mu1 = []
	mu1.append(np.array([0.1, 0.3]))
	mu1.append(np.array([0.2, 0.1]))
	mu1.append(np.array([0.15, 0.4]))
	mu1.append(np.array([0.3, 0.2]))
	
	# This one might be more tricky for pareto based strategies because they cannot drop any arm.
	mu2 = []
	mu2.append(np.array([0.1, 0.3]))
	mu2.append(np.array([0.2, 0.1]))
	mu2.append(np.array([0.08, 0.4]))
	mu2.append(np.array([0.3, 0.08]))

	trial = 4
	noise = gn(1,0,0.01,[-0.2,0.2])
	envs = list()
	envs.append(EnvMultiOutput(num_arm, mu1, noise, weights))
	envs.append(EnvMultiOutput(num_arm, mu2, noise, weights))

	delta = 0.05
	algorithms = list()
	algorithms.append(ParetoUCB(T, num_arm, num_objectives=2, alpha=1, gini_weights=weights))
	algorithms.append(BasicMultiObjective(T, num_arm, num_objectives=2, delta=0.05, gini_weights=weights))

	algorithm_names = []
	algorithm_names.append("ParetoUCB")
	algorithm_names.append("MO_OGDE")
	env_names = []
	env_names.append("favourable env")
	env_names.append("tricky env")

	test(T, trial, envs, algorithms, algorithm_names, env_names)
