import sys
sys.path.append('../')

from helpers.masterTester import test

from environment.envParabola import EnvParabola
from algorithms.bgd import BGD
from algorithms.modular.metaPBGD import MetaPBGD



if __name__ == "__main__":
	T = 1000
	repeats = 10

	envs = []
	envs.append(EnvParabola(dimensions=3, pos_mean=0, pos_scale=5, slope_mean=1, slope_scale=0.5, boundaries=10, stability=0.99, fixed_breakpoints=True))
	envs.append(EnvParabola(dimensions=3, pos_mean=0, pos_scale=5, slope_mean=1, slope_scale=0.5, boundaries=10, stability=1))


	algorithms = []
	algorithms.append(MetaPBGD(T))
	algorithms.append(BGD(T))
	algorithm_names = ["BGD", "MetaBGD"]
	env_names = ["fully stable", "less stable"]

	test(T, repeats, envs, algorithms, algorithm_names, env_names)
