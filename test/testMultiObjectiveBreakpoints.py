from masterTester import test
import sys
sys.path.append('../')

from environment.Gaussian_noise import Gaussian_noise as gn
from environment.multiOutput import EnvMultiOutput
from algorithms.modular.basicMultiObjective import BasicMultiObjective
from algorithms.modular.slidingWindowMO import SlidingWindowMO

import numpy as np

num_arm = 3
T = 200

weights = [1, 1/2]

mu = []
mu.append(np.array([0.1, 0.3]))
mu.append(np.array([0.2, 0.1]))
mu.append(np.array([0.1, 0.4]))

trial = 10
noise = gn(1,0,0.01,[-0.2,0.2])
envs = list()
envs.append(EnvMultiOutput(num_arm, mu, noise, weights))

algorithms = list()
algorithms.append(BasicMultiObjective(T, num_arm, num_objectives=2, delta=0.95, gini_weights=weights))
algorithms.append(SlidingWindowMO(T, num_arm, num_objectives=2, delta=0.95, gini_weights=weights, window_len=20))

algorithm_names = ["MO_OGDE", "With SW"]
env_names = ["Simple"]

test(T, trial, envs, algorithms, algorithm_names, env_names)
