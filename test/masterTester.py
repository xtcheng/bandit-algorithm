import matplotlib.pyplot as plt

def test(T, repeats, envs, algorithms, algorithm_names, env_names):

	avg_regret = []
	cum_regret = []

	for env in envs:
		for i, algorithm in enumerate(algorithms):
			cum_regret.append([0]*T)
			avg_regret.append([0]*T)
			for trial in range(repeats):
				algorithm.clear()
				algorithm.run(env)
				if hasattr(env, "clear"):
					env.clear()
				for y in range(T):
					cum_regret[-1][y] += algorithm.get_cum_rgt()[y] / repeats
					avg_regret[-1][y] += algorithm.get_avg_rgt()[y] / repeats

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
			plt.plot(range(T), avg_regret[len(algorithms)*j + i], label=algorithm_names[i]+" on "+env_names[j])
	plt.xlabel('t (Trials)', fontsize=15)
	plt.ylabel('Average Regret', fontsize=15)
	plt.legend(loc='upper right')
	plt.title('Average Regret')
	plt.show()
