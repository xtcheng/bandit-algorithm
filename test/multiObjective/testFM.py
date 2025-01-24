import sys
sys.path.append('../../')

import helpers.masterTester as tester

from environment.Gaussian_noise import Gaussian_noise as gn
from environment.multiOutput import EnvMultiOutput
from environment.multiOutputBernulli import EnvMultiOutputBernulli
from algorithms.modular.moduleUsers.basicMultiObjective import BasicMultiObjective
from algorithms.modular.moduleUsers.paretoUCB import ParetoUCB

import numpy as np

if __name__ == "__main__":
	mu = np.loadtxt("../../fm_banditified.csv", delimiter=',', encoding="utf-8")
	topstring = ""
	with open("../../fm_banditified.csv", encoding="utf-8") as file:
		for line in file:
			if line[0] == '#':
				topstring += line[1:]
			else:
				# We expect the command at the top, and only at the top.
				break
	print(topstring)
	#print(mu)
	
	num_arm = len(mu)
	num_objectives = len(mu[0])
	T = 3000

	weights = [0]*num_objectives
	for i in range(num_objectives):
		weights[i] = (1/2)**i


	trial = 4
	noise = gn(1,0,0.01,[-0.2,0.2])
	envs = list()
	envs.append(EnvMultiOutput(num_arm, mu, noise, weights))
	envs.append(EnvMultiOutputBernulli(num_arm, mu, weights))

	delta = 0.05
	algorithms = list()
	algorithms.append(ParetoUCB(T, num_arm, num_objectives=num_objectives, alpha=1, gini_weights=weights))
	algorithms.append(BasicMultiObjective(T, num_arm, num_objectives=num_objectives, delta=0.05, gini_weights=weights))

	algorithm_names = []
	algorithm_names.append("ParetoUCB")
	algorithm_names.append("MO_OGDE")
	env_names = []
	env_names.append("transparent_env")
	env_names.append("bernulli_env")

	tester.test(T, trial, envs, algorithms, algorithm_names, env_names)
