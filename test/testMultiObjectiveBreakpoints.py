from masterTester import test
import sys
sys.path.append('../')

from environment.Gaussian_noise import Gaussian_noise as gn
from environment.multiOutput import EnvMultiOutput
from environment.multiOutputNonStationary import EnvMultiOutputNonStationary
from algorithms.modular.basicMultiObjective import BasicMultiObjective
from algorithms.modular.slidingWindowMO import SlidingWindowMO
from algorithms.modular.discountedMO import DiscountedMO

import numpy as np

num_arm = 3
T = 200

weights = [1, 1/2]

mu1 = []
mu1.append(np.array([0.1, 0.3]))
mu1.append(np.array([0.2, 0.1]))
mu1.append(np.array([0.1, 0.4]))

mu2 = []
mu2.append(np.array([0.4, 0.1]))
mu2.append(np.array([0.3, 0.2]))
mu2.append(np.array([0.1, 0.4]))

"""mu2 = []
mu2.append(np.array([0.1, 0.3]))
mu2.append(np.array([0.1, 0.4]))
mu2.append(np.array([0.2, 0.1]))"""

mu3 = []
mu3.append(np.array([0.1, 0.3]))
mu3.append(np.array([0.2, 0.1]))
mu3.append(np.array([0.1, 0.4]))

trial = 10
noise = gn(1,0,0.01,[-0.2,0.2])
envs = list()
envs.append(EnvMultiOutput(num_arm, mu1, noise, weights))
envs.append(EnvMultiOutputNonStationary(num_arm, [mu1, mu2, mu3], noise, weights, [100, 150]))

algorithms = list()
algorithms.append(BasicMultiObjective(T, num_arm, num_objectives=2, delta=0.95, gini_weights=weights))
algorithms.append(SlidingWindowMO(T, num_arm, num_objectives=2, delta=0.95, gini_weights=weights, window_len=20))
algorithms.append(DiscountedMO(T, num_arm, num_objectives=2, delta=0.95, gini_weights=weights, gamma=0.9))

algorithm_names = ["MO_OGDE", "SlidingWindow", "Discounted"]
env_names = ["Simple", "withBreakpoints"]

test(T, trial, envs, algorithms, algorithm_names, env_names)
