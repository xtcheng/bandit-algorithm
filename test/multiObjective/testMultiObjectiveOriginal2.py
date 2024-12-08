import sys
sys.path.append('../../')

from helpers.masterTester import test

from environment.Gaussian_noise import Gaussian_noise as gn
from environment.multiOutputRandomized import EnvMultiOutputRandomized
from algorithms.modular.moduleUsers.basicMultiObjective import BasicMultiObjective
from algorithms.modular.moduleUsers.deltalessMultiObjective import DeltalessMultiObjective

import numpy as np

if __name__ == "__main__":
	num_arm = 5
	num_objectives=5

	T = 10**4

	weights = []
	for i in range(0, 20):
		weights.append(1/(2**i))

	trial = 64
	envs = list()
	envs.append(EnvMultiOutputRandomized(num_arm, num_objectives, weights))

	algorithms = list()
	algorithms.append(DeltalessMultiObjective(T, num_arm=num_arm, num_objectives=num_objectives, gini_weights=weights))
	algorithms.append(BasicMultiObjective(T, num_arm=num_arm, num_objectives=num_objectives, delta=0.05, gini_weights=weights))

	algorithm_names = ["Deltaless MO_OGDE", "MO_OGDE"]
	env_names = ["random environment"]

	test(T, trial, envs, algorithms, algorithm_names, env_names, True)
