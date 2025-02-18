import sys
sys.path.append('../../')

from helpers.masterTester import test

from environment.envParabola import EnvParabola
from algorithms.bgd import BGD
from algorithms.gb_oco_ltc import GB_OCO_LTC
#from algorithms.modular.metaPBGD import MetaPBGD



if __name__ == "__main__":
	T = 1000
	repeats = 1
	
	constrainted_env = EnvParabola(dimensions=3, pos_mean=-0.5, pos_scale=0.2, slope_mean=1, slope_scale=0, boundaries=10, stability=1)
	vars = constrainted_env.getVariables()
	constraint1 = vars[0] + vars[1] + vars[2] - 1
	constrainted_env.addConstraint(constraint1)
	
	envs = []
	envs.append(constrainted_env)
	
	
	algorithms = []
	#algorithms.append(MetaPBGD(T))
	algorithms.append(GB_OCO_LTC(T, 0.05, 1))
	algorithm_names = ["test"]
	env_names = ["env"]
	
	test(T, repeats, envs, algorithms, algorithm_names, env_names)
