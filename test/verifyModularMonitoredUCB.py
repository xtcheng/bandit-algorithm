import sys
sys.path.append('../')

from helpers.masterTester import test

from environment.Gaussian_noise import Gaussian_noise as gn
from environment.envNonStationary import env_non_stationary
from algorithms.monitoredUCB import MonitoredUCB
from algorithms.modular.moduleUsers.monitoredUCB import MonitoredUCB as MonitoredUCBModular

import numpy as np

num_arm = 3
T = 10000

# Stochastic
mu1 = np.array([0.5, 0.3, 0.4])
mu2 = np.array([0.5, 0.3, 0.9])
mu3 = np.array([0.5, 0.3, 0.4])
Trial = 10
noise = gn(1,0,0.01,[-0.2,0.2])
envs = list()
envs.append(env_non_stationary(num_arm, [mu1, mu2, mu3], noise, [3000, 5000]))


algorithms = list()
algorithms.append(MonitoredUCB(T, num_arm, w=50, b=3, gamma=0.1))
algorithms.append(MonitoredUCBModular(T, num_arm, w=50, b=3, gamma=0.1))

algorithm_names = ["Old", "Modular"]
env_names = ["non stationary"]

test(T, Trial, envs, algorithms, algorithm_names, env_names)
# Basically the same results, but:
#Average time for Old: 1.102609669999947 seconds.
#Average time for Modular: 1.1457539999999427 seconds.

# Open issue, low priority: Why is the modular version notably slower? (Object-orientation overhead or something avoidable?)
