from masterTester import test
import sys
sys.path.append('../')

from environment.envParabola import EnvParabola
from algorithms.greedyProjection import GreedyProjection
from algorithms.bgd import BGD
from algorithms.bgdOld import BGD2



T = 500
repeats = 30

envs = list()
envs.append(EnvParabola(dimensions=3, pos_mean=0, pos_scale=5, slope_mean=1, slope_scale=0.5, boundaries=10, stability=1))
envs.append(EnvParabola(dimensions=3, pos_mean=0, pos_scale=5, slope_mean=1, slope_scale=0.5, boundaries=10, stability=0.7))
envs.append(EnvParabola(dimensions=3, pos_mean=0, pos_scale=5, slope_mean=1, slope_scale=0.5, boundaries=10, stability=0))


algorithms = [BGD2(T), BGD(T)]
algorithm_names = ["Unfixed BGD", "Fixed BGD"]
env_names = ["fully stable", "less stable", "unstable"]

test(T, repeats, envs, algorithms, algorithm_names, env_names)
