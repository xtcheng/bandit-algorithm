import sys
sys.path.append('../../')

import helpers.masterTester as tester



from environment.Gaussian_noise import Gaussian_noise as gn
from environment.multiOutputNonStationary import EnvMultiOutputNonStationary
from algorithms.modular.moduleUsers.basicMultiObjective import BasicMultiObjective
from algorithms.modular.moduleUsers.discountedMO import DiscountedMO
from algorithms.modular.moduleUsers.monitoredMO import MonitoredMO
from algorithms.modular.moduleUsers.bocdMO import BOCD_MO
from algorithms.modular.moduleUsers.expertsMultiObjective import ExpertsMultiObjective
from algorithms.modular.moduleUsers.discountedExpertsMultiObjective import DiscountedExpertsMultiObjective
from algorithms.modular.moduleUsers.bogdExpertsMultiObjective import BOCD_ExpertsMultiObjective

import numpy as np


# Compare the normal Meta strategy against ones that has breakpoint adaption against ones that have both.
# Note that this environment is a little mean, especially in the second half.
if __name__ == "__main__":
	num_arm = 3
	num_objectives = 3
	T = 3000

	weights = [1, 1/2, 1/4]

	mu1 = []
	mu1.append(np.array([0.1, 0.3, 0.2]))
	mu1.append(np.array([0.2, 0.1, 0.2]))
	mu1.append(np.array([0.1, 0.4, 0.3]))

	mu2 = []
	mu2.append(np.array([0.1, 0.3, 0.1]))
	mu2.append(np.array([0.1, 0.4, 0.2]))
	mu2.append(np.array([0.2, 0.1, 0.1]))

	mu3 = []
	mu3.append(np.array([0.2, 0.1, 0.2]))
	mu3.append(np.array([0.1, 0.4, 0.3]))
	mu3.append(np.array([0.1, 0.3, 0.2]))

	mu4 = []
	mu4.append(np.array([0.1, 0.3, 0.2]))
	mu4.append(np.array([0.2, 0.1, 0.2]))
	mu4.append(np.array([0.1, 0.4, 0.3]))

	mu5 = []
	mu5.append(np.array([0.1, 0.3, 0.2]))
	mu5.append(np.array([0.2, 0.1, 0.2]))
	mu5.append(np.array([0.1, 0.4, 0.3]))

	trial = 16
	noise = gn(1,0,0.01,[-0.2,0.2])
	envs = list()
	envs.append(EnvMultiOutputNonStationary(num_arm, [mu1, mu2, mu3, mu4, mu5], noise, weights, [700, 1500, 1700, 2300]))

	delta = 0.05
	algorithms = list()
	algorithms.append(DiscountedMO(T, num_arm, num_objectives=num_objectives, delta=delta, gini_weights=weights, gamma=0.99))
	#algorithms.append(MonitoredMO(T, num_arm, num_objectives=num_objectives, delta=delta, gini_weights=weights, w=50, b=3))
	algorithms.append(BOCD_MO(T, num_arm, num_objectives=num_objectives, delta=delta, gini_weights=weights))
	algorithms.append(BasicMultiObjective(T, num_arm=num_arm, num_objectives=num_objectives, delta=0.05, gini_weights=weights))
	algorithms.append(ExpertsMultiObjective(T, num_arm=num_arm, num_objectives=num_objectives, gini_weights=weights))
	algorithms.append(DiscountedExpertsMultiObjective(T, num_arm=num_arm, num_objectives=num_objectives, gini_weights=weights, gamma=0.99))
	algorithms.append(BOCD_ExpertsMultiObjective(T, num_arm=num_arm, num_objectives=num_objectives, gini_weights=weights))

	algorithm_names = []
	algorithm_names.append("Discounted")
	#algorithm_names.append("Monitored")
	algorithm_names.append("BOCD")
	algorithm_names.append("Original")
	algorithm_names.append("Meta")
	algorithm_names.append("Discounted_Meta")
	algorithm_names.append("BOGD_Meta")
	env_names = []
	env_names.append("Breakpoint env")
	
	tester.resultpath = "metaBreakpointsResults/"
	
	tester.purgeResults()
	tester.testOnly(T, trial, envs, algorithms, algorithm_names, env_names)
	tester.readAllResults()
	tester.plotResults()
