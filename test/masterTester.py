import matplotlib.pyplot as plt
import time

def test(T, repeats, envs, algorithms, algorithm_names, env_names):

	avg_regret = []
	cum_regret = []
	psd_regret1 = []
	psd_regret2 = []
	regret1 = []
	regret2 = []
	sum_times = [0]*len(algorithms)
	
	assert len(envs) == len(env_names) and len(algorithms) == len(algorithm_names)
	
	for env in envs:
		for i, algorithm in enumerate(algorithms):
			cum_regret.append([0]*T)
			avg_regret.append([0]*T)
			psd_regret1.append([0]*T)
			psd_regret2.append([0]*T)
			regret1.append([0]*T)
			regret2.append([0]*T)
			start_time = time.perf_counter()
			for trial in range(repeats):
				algorithm.clear()
				algorithm.run(env)
				if hasattr(env, "clear"):
					env.clear()
				for y in range(T):
					cum_regret[-1][y] += algorithm.get_cum_rgt()[y] / repeats
					avg_regret[-1][y] += algorithm.get_avg_rgt()[y] / repeats
					psd_regret1[-1][y] += algorithm.get_psd_rgt1()[y] / repeats
					psd_regret2[-1][y] += algorithm.get_psd_rgt2()[y] / repeats
					regret1[-1][y] += algorithm.get_rgt1()[y] / repeats
					regret2[-1][y] += algorithm.get_rgt2()[y] / repeats
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
	
	
	plt.figure(figsize=(6, 5))
	for j, env in enumerate(envs):
		for i, algorithm in enumerate(algorithms):
			plt.plot(range(T), psd_regret1[len(algorithms)*j + i], label=algorithm_names[i]+" on "+env_names[j])
	plt.xlabel('t (Trials)', fontsize=15)
	plt.ylabel('Pseudo Regret 1', fontsize=15)
	plt.legend(loc='upper right')
	plt.title('Pseudo Regret 1')
	
	plt.figure(figsize=(6, 5))
	for j, env in enumerate(envs):
		for i, algorithm in enumerate(algorithms):
			plt.plot(range(T), psd_regret2[len(algorithms)*j + i], label=algorithm_names[i]+" on "+env_names[j])
	plt.xlabel('t (Trials)', fontsize=15)
	plt.ylabel('Pseudo Regret 2', fontsize=15)
	plt.legend(loc='upper right')
	plt.title('Pseudo Regret 2')
	
	
	plt.figure(figsize=(6, 5))
	for j, env in enumerate(envs):
		for i, algorithm in enumerate(algorithms):
			plt.plot(range(T), avg_regret[len(algorithms)*j + i], label=algorithm_names[i]+" on "+env_names[j])
	plt.xlabel('t (Trials)', fontsize=15)
	plt.ylabel('Average Regret', fontsize=15)
	plt.legend(loc='upper right')
	plt.title('Average Regret')
	plt.show()
	
	
	plt.figure(figsize=(6, 5))
	for j, env in enumerate(envs):
		for i, algorithm in enumerate(algorithms):
			plt.plot(range(T), regret1[len(algorithms)*j + i], label=algorithm_names[i]+" on "+env_names[j])
	plt.xlabel('t (Trials)', fontsize=15)
	plt.ylabel('\"Regret\" 1', fontsize=15)
	plt.legend(loc='upper right')
	plt.title('\"Regret\" 1')
	plt.show()
	
	
	plt.figure(figsize=(6, 5))
	for j, env in enumerate(envs):
		for i, algorithm in enumerate(algorithms):
			plt.plot(range(T), regret2[len(algorithms)*j + i], label=algorithm_names[i]+" on "+env_names[j])
	plt.xlabel('t (Trials)', fontsize=15)
	plt.ylabel('\"Regret\" 2', fontsize=15)
	plt.legend(loc='upper right')
	plt.title('\"Regret\" 2')
	plt.show()
