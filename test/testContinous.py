import sys
sys.path.append('../')

from helpers.masterTester import test

from environment.envParabola import EnvParabola
from algorithms.greedyProjection import GreedyProjection
from algorithms.bgd import BGD



T = 100
repeats = 10

envs = list()
envs.append(EnvParabola(dimensions=3, pos_mean=0, pos_scale=5, slope_mean=1, slope_scale=0.5, boundaries=10, stability=1))
envs.append(EnvParabola(dimensions=3, pos_mean=0, pos_scale=5, slope_mean=1, slope_scale=0.5, boundaries=10, stability=0.3))
envs.append(EnvParabola(dimensions=3, pos_mean=0, pos_scale=5, slope_mean=1, slope_scale=0.5, boundaries=10, stability=0))


algorithms = [GreedyProjection(T), BGD(T)]
algorithm_names = ["Greedy Projection", "BGD"]
env_names = ["fully stable", "less stable", "unstable"]

test(T, repeats, envs, algorithms, algorithm_names, env_names)
