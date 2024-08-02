from masterTester import test
import sys
sys.path.append('../')

from environment.Gaussian_noise import Gaussian_noise as gn
from environment.env import env_stochastic
from environment.envNonStationary import env_non_stationary
from algorithms.UCB1 import UCB1
from algorithms.slidingWindowUCB import SlidingWindowUCB
from algorithms.monitoredUCB import MonitoredUCB
from algorithms.modular.glr_klUCB import GLR_klUCB

import numpy as np

num_arm = 3
T = 10000

# Stochastic
mu1 = np.array([0.5, 0.3, 0.4])
mu2 = np.array([0.5, 0.3, 0.9])
mu3 = np.array([0.5, 0.3, 0.4])
Trial = 1
noise = gn(1,0,0.01,[-0.2,0.2])
envs = list()
envs.append(env_non_stationary(num_arm, [mu1, mu2, mu3], noise, [3000, 5000]))

# tuning for
# T=10000, M=3, v(i+1)-v(i) >= 2000, delta(k,i) = 0.5
# So 3 < floor(10000/L) and 2000 > L
# and 0.5 >= 2sqrt(log(2KT^2)/w) + 2sqrt(log(2T)/w)
# => 0.5 >= 2(2.881 + 2.074)/sqrt(w) => 0.5*sqrt(w) >= 9.91

# One solution: w = 50, L=50*ceil(3/gamma)
# gamma=0.1
#406,0.65

algorithms = list()
algorithms.append(UCB1(T, num_arm, xi=0.5))
algorithms.append(SlidingWindowUCB(T, num_arm, 800, xi=0.5))
algorithms.append(MonitoredUCB(T, num_arm, w=50, b=3, gamma=0.1))
algorithms.append(GLR_klUCB(T, num_arm, alpha=0.03, delta=0.01, global_restart=True))
algorithms.append(GLR_klUCB(T, num_arm, alpha=0.03, delta=0.01, global_restart=False))

algorithm_names = ["UCB1", "SlidingWindowUCB", "MonitoredUCB", "GLR-klUCB glob", "GLR-klUCB loc"]
env_names = ["non stationary"]

test(T, Trial, envs, algorithms, algorithm_names, env_names)
