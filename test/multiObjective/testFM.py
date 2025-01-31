import sys
sys.path.append('../../')

import helpers.masterTester as tester

from environment.Gaussian_noise import Gaussian_noise as gn
from environment.multiOutput import EnvMultiOutput
from environment.multiOutputBernulli import EnvMultiOutputBernulli
from environment.multiOutputNonStationary import EnvMultiOutputNonStationary
from algorithms.modular.moduleUsers.basicMultiObjective import BasicMultiObjective
from algorithms.modular.moduleUsers.expertsMultiObjective import ExpertsMultiObjective
from algorithms.modular.moduleUsers.paretoUCB import ParetoUCB

import numpy as np

if __name__ == "__main__":
	mu = [0]*4
	calls = [0]*4
	for dayphase in range(4):
		if dayphase == 0:
			filename = "../../fm_banditified.csv"
		elif dayphase == 1:
			filename = "../../fm_banditified_day.csv"
		elif dayphase == 2:
			filename = "../../fm_banditified_evening.csv"
		else:
			filename = "../../fm_banditified_night.csv"
		
		mu[dayphase] = np.loadtxt(filename, delimiter=',', encoding="utf-8")
		topstring = ""
		with open(filename, encoding="utf-8") as file:
			for line in file:
				if line[0] == '#':
					topstring += line[1:]
					if "calls=" in line:
						calls[dayphase] = int(line.strip().split('=')[1])
				else:
					# We expect the comment at the top, and only at the top.
					break
		print(topstring)
		#print(mu[dayphase])
	assert calls[0] == sum(calls[1:]), "calls in dayphases do not sum up to "+str(calls[0])
	
	num_arm = len(mu[0])
	num_objectives = len(mu[0][0])
	T = 3000
	
	factor = T/calls[0]
	breakpoint1 = round((calls[1] + 1)*factor)
	breakpoint2 = round((calls[1] + calls[2]+1)*factor)
	#print("Breakpoints:", breakpoint1, breakpoint2)
	
	weights = [0]*num_objectives
	for i in range(num_objectives):
		weights[i] = (1/2)**i


	trial = 4
	noise = gn(1,0,0.01,[-0.2,0.2])
	envs = list()
	envs.append(EnvMultiOutput(num_arm, mu[0], noise, weights))
	envs.append(EnvMultiOutputBernulli(num_arm, mu[0], weights))
	envs.append(EnvMultiOutputNonStationary(num_arm, mu[1:], noise, weights, [breakpoint1, breakpoint2]))

	delta = 0.05
	algorithms = list()
	algorithms.append(ParetoUCB(T, num_arm, num_objectives=num_objectives, alpha=1, gini_weights=weights))
	#algorithms.append(BasicMultiObjective(T, num_arm, num_objectives=num_objectives, delta=0.05, gini_weights=weights))
	algorithms.append(ExpertsMultiObjective(T, num_arm, num_objectives=num_objectives, gini_weights=weights))

	algorithm_names = []
	algorithm_names.append("ParetoUCB")
	#algorithm_names.append("MO_OGDE")
	algorithm_names.append("Meta_MO_OGDE")
	env_names = []
	env_names.append("transparent_env")
	env_names.append("bernoulli_env")
	env_names.append("transparent_dayphase_env")

	tester.test(T, trial, envs, algorithms, algorithm_names, env_names)
