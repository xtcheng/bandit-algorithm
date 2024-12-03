import sys
sys.path.append('../../')

from helpers.masterTester import test

from environment.Gaussian_noise import Gaussian_noise as gn
from environment.multiOutput import EnvMultiOutput
from environment.multiOutputNonStationary import EnvMultiOutputNonStationary
from algorithms.modular.moduleUsers.basicMultiObjective import BasicMultiObjective
from algorithms.modular.moduleUsers.slidingWindowMO import SlidingWindowMO
from algorithms.modular.moduleUsers.discountedMO import DiscountedMO
from algorithms.modular.moduleUsers.monitoredMO import MonitoredMO
from algorithms.modular.moduleUsers.bocdMO import BOCD_MO
from algorithms.modular.moduleUsers.glrMO import GLR_MO

import numpy as np

if __name__ == "__main__":
	num_arm = 3
	T = 10000

	weights = [1, 1/2]

	mu1 = []
	mu1.append(np.array([0.1, 0.3]))
	mu1.append(np.array([0.2, 0.1]))
	mu1.append(np.array([0.1, 0.4]))

	mu2 = []
	mu2.append(np.array([0.1, 0.3]))
	mu2.append(np.array([0.1, 0.4]))
	mu2.append(np.array([0.2, 0.1]))

	mu3 = []
	mu3.append(np.array([0.1, 0.3]))
	mu3.append(np.array([0.2, 0.1]))
	mu3.append(np.array([0.1, 0.4]))

	trial = 16
	noise = gn(1,0,0.01,[-0.2,0.2])
	envs = list()
	envs.append(EnvMultiOutputNonStationary(num_arm, [mu1, mu2, mu3], noise, weights, [3000, 5000]))

	delta = 0.05
	algorithms = list()
	algorithms.append(BasicMultiObjective(T, num_arm, num_objectives=2, delta=delta, gini_weights=weights))
	algorithms.append(SlidingWindowMO(T, num_arm, num_objectives=2, delta=delta, gini_weights=weights, window_len=800))
	algorithms.append(DiscountedMO(T, num_arm, num_objectives=2, delta=delta, gini_weights=weights, gamma=0.9975))
	algorithms.append(MonitoredMO(T, num_arm, num_objectives=2, delta=delta, gini_weights=weights, w=50, b=3))
	algorithms.append(BOCD_MO(T, num_arm, num_objectives=2, delta=delta, gini_weights=weights))
	algorithms.append(GLR_MO(T, num_arm, num_objectives=2, delta=delta, delta2=0.01, global_restart=True, lazyness=10, gini_weights=weights))

	algorithm_names = []
	algorithm_names.append("MO_OGDE")
	algorithm_names.append("SlidingWindow")
	algorithm_names.append("Discounted")
	algorithm_names.append("Monitored")
	algorithm_names.append("BOCD_MO")
	algorithm_names.append("GLR_MO")
	env_names = []
	env_names.append("withBreakpoints")

	test(T, trial, envs, algorithms, algorithm_names, env_names)
