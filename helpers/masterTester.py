import matplotlib.pyplot as plt
import time

def test(T, repeats, envs, algorithms, algorithm_names, env_names):

	avg_regret = []
	cum_regret = []
	psd_regret = []
	has_pseudo = False
	sum_times = [0]*len(algorithms)
	
	assert len(envs) == len(env_names) and len(algorithms) == len(algorithm_names)
	
	for env in envs:
		for i, algorithm in enumerate(algorithms):
			cum_regret.append([0]*T)
			avg_regret.append([0]*T)
			psd_regret.append([0]*T)
			start_time = time.perf_counter()
			for trial in range(repeats):
				algorithm.clear()
				algorithm.run(env)
				if hasattr(env, "clear"):
					env.clear()
				for y in range(T):
					cum_regret[-1][y] += algorithm.get_cum_rgt()[y] / repeats
					avg_regret[-1][y] += algorithm.get_avg_rgt()[y] / repeats
					if hasattr(algorithm, "get_psd_rgt"):
						psd_regret[-1][y] += algorithm.get_psd_rgt()[y] / repeats
						has_pseudo = True
			end_time = time.perf_counter()
			sum_times[i] += end_time - start_time
	
	for i in range(len(algorithms)):
		print("Average time for "+algorithm_names[i]+": "+str(sum_times[i] / (repeats*len(envs)))+" seconds.")
	
	plt.figure(figsize=(6, 5))
	for j, env in enumerate(envs):
		for i, algorithm in enumerate(algorithms):
			plt.plot(range(T), cum_regret[len(algorithms)*j + i], label=algorithm_names[i]+" on "+env_names[j])
	plt.xlabel('t (Trials)', fontsize=15)
	plt.ylabel('Cumulative Regret', fontsize=15)
	plt.legend(loc='upper right')
	plt.title('Cumulative Regret')
	#plt.show()
	
	
	if has_pseudo:
		plt.figure(figsize=(6, 5))
		for j, env in enumerate(envs):
			for i, algorithm in enumerate(algorithms):
				plt.plot(range(T), psd_regret[len(algorithms)*j + i], label=algorithm_names[i]+" on "+env_names[j])
		plt.xlabel('t (Trials)', fontsize=15)
		plt.ylabel('Pseudo Regret', fontsize=15)
		plt.legend(loc='upper right')
		plt.title('Pseudo Regret')
	
	
	plt.figure(figsize=(6, 5))
	for j, env in enumerate(envs):
		for i, algorithm in enumerate(algorithms):
			plt.plot(range(T), avg_regret[len(algorithms)*j + i], label=algorithm_names[i]+" on "+env_names[j])
	plt.xlabel('t (Trials)', fontsize=15)
	plt.ylabel('Average Regret', fontsize=15)
	plt.legend(loc='upper right')
	plt.title('Average Regret')
	plt.show()
