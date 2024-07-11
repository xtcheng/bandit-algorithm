from masterTester import test
import sys
sys.path.append('../')

from environment.Gaussian_noise import Gaussian_noise as gn
from environment.env import env_stochastic
from algorithms.UCB1 import UCB1
from algorithms.discountedUCB import DiscountedUCB
from algorithms.slidingWindowUCB import SlidingWindowUCB

import numpy as np

num_arm = 5
T = 5000

# Stochastic
mu = np.array([0.1, 0.8, 0.4, 0.3, 0.6])
Trial = 8
noise = gn(1,0,0.01,[-0.2,0.2])
envs = list()
envs.append(env_stochastic(num_arm, mu, noise))
envs.append(env_stochastic(num_arm, mu, noise, stability=0.996))# About 1 change in 50 turns.


algorithms = [UCB1(T, num_arm), DiscountedUCB(T, num_arm, 0.99), SlidingWindowUCB(T, num_arm, 100)]
algorithm_names = ["UCB1", "DiscountedUCB", "SlidingWindowUCB"]
env_names = ["stochastic", "non stationary"]

test(T, Trial, envs, algorithms, algorithm_names, env_names)
