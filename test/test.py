import sys
sys.path.append('../')

import helpers.masterTester as tester
from environment.Gaussian_noise import Gaussian_noise as gn
import numpy as np

tester.ERRORBAR_COUNT = 40
tester.AVAILABLE_CORES = 8

#from environment.envContextual import EnvContextual
from environment.envcont import environment as EnvContextual
from algorithms.UCB1 import UCB1
from algorithms.modular.moduleUsers.linUCB import LinUCB
from algorithms.modular.moduleUsers.linUCB2 import LinUCB2

if __name__ == '__main__':
	num_arm = 7
	T = 1000

	# Stochastic
	user_features = [0.5, 0.3, 0.4, 0.2, 0.9, 0]
	Trial = 1
	noise = gn(1,0,0.01,[-0.2,0.2])
	envs = list()
	envs.append(EnvContextual(num_arm, len(user_features), user_features, 1))
	#envs.append(EnvContextual(num_arm, user_features, noise, False))
	#envs.append(EnvContextual(num_arm, user_features, noise, True))


	algorithms = []
	algorithms.append(LinUCB(T, num_arm, len(user_features), alpha=0.5))
	algorithms.append(LinUCB2(T, num_arm, len(user_features), alpha=0.5))
	#algorithms.append(UCB1(T, num_arm, xi=0.5))
    
	algorithm_names = []
	algorithm_names.append("LinUCB")
	#algorithm_names.append("UCB1")
	algorithm_names.append("LinUCB variant")
	env_names = []
	env_names.append("contextual")
	#env_names.append("contextual_renewing")

	print(algorithm_names)
	tester.test(T, Trial, envs, algorithms, algorithm_names, env_names)