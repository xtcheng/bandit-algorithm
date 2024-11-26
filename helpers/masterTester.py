import matplotlib.pyplot as plt
import time
import numpy as np
import math

def evaluate(samples, index2):
	temp_sum = 0
	for trial in range(len(samples)):
		temp_sum += samples[trial][index2]
	avg = temp_sum / len(samples)
	
	temp_sum = 0
	for trial in range(len(samples)):
		temp_sum += math.sqrt((samples[trial][index2] - avg)**2)
	var = temp_sum / len(samples)
	return avg, var

def makeInterval(average, variance):
	average = np.array(average)
	variance = np.array(variance)
	return average-variance, average+variance

def plotOnce(env_names, algorithm_names, samples, samples_var, label, logscale):
	plt.figure(figsize=(6, 5))
	for j, env in enumerate(env_names):
		for i, algorithm in enumerate(algorithm_names):
			plt.plot(range(len(samples[0])), samples[len(algorithm_names)*j + i], label=algorithm_names[i]+" on "+env_names[j])
			var_low, var_up = makeInterval(samples[len(algorithm_names)*j + i], samples_var[len(algorithm_names)*j + i])
			plt.fill_between(range(len(samples[0])), var_low, var_up, color="xkcd:light grey")
		
	plt.xlabel('t (Trials)', fontsize=15)
	plt.ylabel(label, fontsize=15)
	plt.legend(loc='upper right')
	plt.title(label)
	if logscale:
		plt.xscale('log')

def test(T, repeats, envs, algorithms, algorithm_names, env_names, logscale=False):

	avg_regret = []
	avg_regret_var = []
	cum_regret = []
	cum_regret_var = []
	psd_regret = []
	psd_regret_var = []
	has_pseudo = False
	sum_times = [0]*len(algorithms)
	
	assert len(envs) == len(env_names) and len(algorithms) == len(algorithm_names)
	
	for env in envs:
		for i, algorithm in enumerate(algorithms):
			raw_cum = [0]*repeats
			raw_avg = [0]*repeats
			raw_psd = [0]*repeats
			cum_regret.append([0]*T)
			avg_regret.append([0]*T)
			psd_regret.append([0]*T)
			cum_regret_var.append([0]*T)
			avg_regret_var.append([0]*T)
			psd_regret_var.append([0]*T)
			start_time = time.perf_counter()
			for trial in range(repeats):
				algorithm.clear()
				algorithm.run(env)
				if hasattr(env, "clear"):
					env.clear()
				raw_cum[trial] = algorithm.get_cum_rgt()
				raw_avg[trial] = algorithm.get_avg_rgt()
				if hasattr(algorithm, "get_psd_rgt"):
					raw_psd[trial] = algorithm.get_psd_rgt()
					has_pseudo = True
			end_time = time.perf_counter()
			sum_times[i] += end_time - start_time
			for y in range(T):
				cum_regret[-1][y], cum_regret_var[-1][y] = evaluate(raw_cum, y)
				avg_regret[-1][y], avg_regret_var[-1][y] = evaluate(raw_avg, y)
				if hasattr(algorithm, "get_psd_rgt"):
					psd_regret[-1][y], psd_regret_var[-1][y] = evaluate(raw_psd, y)
	
	for i in range(len(algorithms)):
		print("Average time for "+algorithm_names[i]+": "+str(sum_times[i] / (repeats*len(envs)))+" seconds.")
	
	plotOnce(env_names, algorithm_names, cum_regret, cum_regret_var, "Cumulative Regret", logscale)
	if has_pseudo:
		plotOnce(env_names, algorithm_names, psd_regret, psd_regret_var, "Pseudo Regret", logscale)
	plotOnce(env_names, algorithm_names, avg_regret, avg_regret_var, "Average Regret", logscale)
	
	plt.show()
