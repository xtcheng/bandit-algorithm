from masterTester import test
import sys
sys.path.append('../')

from environment.Gaussian_noise import Gaussian_noise as gn
from environment.env import env_stochastic
from environment.envNonStationary import env_non_stationary
from algorithms.monitoredUCB import MonitoredUCB
from algorithms.monitoredUCBOriginal import MonitoredUCBOriginal

import numpy as np

num_arm = 3
T = 10000

# Stochastic
mu1 = np.array([0.5, 0.3, 0.4])
mu2 = np.array([0.5, 0.3, 0.9])
mu3 = np.array([0.5, 0.3, 0.4])
Trial = 4
noise = gn(1,0,0.01,[-0.2,0.2])
envs = list()
envs.append(env_non_stationary(num_arm, [mu1, mu2, mu3], noise, [3000, 5000]))

algorithms = [MonitoredUCB(T, num_arm, w=50, b=3, gamma=0.1), MonitoredUCBOriginal(T, num_arm, w=50, b=3, gamma=0.1)]
algorithm_names = ["MonitoredUCB", "MonitoredUCBOriginal"]
env_names = ["non stationary"]

test(T, Trial, envs, algorithms, algorithm_names, env_names)
