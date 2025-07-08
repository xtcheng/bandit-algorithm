import sys
sys.path.append('../')

import helpers.masterTester as tester
from environment.Gaussian_noise import Gaussian_noise as gn
import numpy as np

tester.ERRORBAR_COUNT = 40
tester.AVAILABLE_CORES = 8

from environment.envContextual import EnvContextual
from environment.envContextualFlip import EnvContextualFlip
from algorithms.modular.moduleUsers.linUCB import LinUCB
from algorithms.modular.moduleUsers.CW_OFUL import CW_OFUL

if __name__ == '__main__':
	num_arm = 7
	T = 1000

	# Stochastic
	user_features = [0.5, 0.3, 0.4, 0.2, 0.9, 0]
	Trial = 8
	noise = gn(1,0,0.01,[-0.2,0.2])
	envs = list()
	#envs.append(EnvContextual(num_arm, len(user_features), user_features, 1))
	envs.append(EnvContextual(num_arm, user_features, noise, False))
	envs.append(EnvContextual(num_arm, user_features, noise, True))
	envs.append(EnvContextualFlip(num_arm, user_features, noise, True, 20))


	delta = 0.05
	alpha = np.sqrt(np.log(2*T*delta) / 2)
	algorithms = []
	algorithms.append(LinUCB(T, num_arm, len(user_features), alpha))
	algorithms.append(CW_OFUL(T, num_arm, len(user_features), 1, alpha, 1))
	
	algorithm_names = []
	algorithm_names.append("LinUCB")
	algorithm_names.append("CW_OFUL")
	env_names = []
	env_names.append("contextual")
	env_names.append("contextual_renewing")
	env_names.append("contextual_renewing_flip")

	print(algorithm_names)
	tester.test(T, Trial, envs, algorithms, algorithm_names, env_names)