import sys
sys.path.append('../')

from helpers.masterTester import test

from environment.Gaussian_noise import Gaussian_noise as gn
from environment.env import env_stochastic
from environment.envNonStationary import env_non_stationary
from algorithms.UCB1 import UCB1
from algorithms.discountedUCB import DiscountedUCB
from algorithms.slidingWindowUCB import SlidingWindowUCB

import numpy as np

num_arm = 3
T = 10000

# Stochastic
mu1 = np.array([0.5, 0.3, 0.4])
mu2 = np.array([0.5, 0.3, 0.9])
mu3 = np.array([0.5, 0.3, 0.4])
Trial = 8
noise = gn(1,0,0.01,[-0.2,0.2])
envs = list()
envs.append(env_stochastic(num_arm, mu1, noise))
envs.append(env_non_stationary(num_arm, [mu1, mu2, mu3], noise, [3000, 5000]))


algorithms = [UCB1(T, num_arm, xi=0.5), DiscountedUCB(T, num_arm, 0.9975, xi=0.5), SlidingWindowUCB(T, num_arm, 800, xi=0.5)]
algorithm_names = ["UCB1", "DiscountedUCB", "SlidingWindowUCB"]
env_names = ["stochastic", "non stationary"]

test(T, Trial, envs, algorithms, algorithm_names, env_names)
