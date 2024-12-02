import matplotlib.pyplot as plt
import time
import numpy as np
import math
import multiprocessing
from copy import deepcopy

def evaluate(samples):
	temp_sum = 0
	for trial in range(len(samples)):
		temp_sum += samples[trial]
	avg = temp_sum / len(samples)
	
	temp_sum = 0
	for trial in range(len(samples)):
		temp_sum += math.sqrt((samples[trial] - avg)**2)
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


def refine(samples, timesteps, rpts):
	ret = [0]*timesteps
	for i in range(timesteps):
		ret[i] = list()
	
	for x in range(rpts):
		one_series = samples.get() # timesteps for ONE of the repetitions. get will block until the data is available.
		for i in range(timesteps):
			ret[i].append(float(one_series[i])) # cast to float to ensure it is not some weird numpy object that would cause trouble in the plotting.
	return ret

def run(algorithm, env, raw_cum, raw_avg, raw_psd, todo):
	for x in range(todo):
		algorithm.run(env)
		raw_cum.put(algorithm.get_cum_rgt())
		raw_avg.put(algorithm.get_avg_rgt())
		if hasattr(algorithm, "get_psd_rgt"):
			raw_psd.put(algorithm.get_psd_rgt())
		
		algorithm.clear()
		if hasattr(env, "clear"):
			env.clear()


def test(T, rpts, envs, algorithms, algorithm_names, env_names, logscale=False):

	avg_regret = []
	avg_regret_var = []
	cum_regret = []
	cum_regret_var = []
	psd_regret = []
	psd_regret_var = []
	has_pseudo = False
	#sum_times = [0]*len(algorithms)
	
	AVAILABLE_CORES = 8 # How many subprocesses will be spawned at maximum. This number should not exceed the number of physical cores available on your system to avoid diminishing returns and crashes! If less repetitions are wanted, only that much processes will be spawned. If more repetitions are wanted, they will be equally distributed among the processes.
	
	assert len(envs) == len(env_names) and len(algorithms) == len(algorithm_names)
	
	for env in envs:
		for i, algorithm in enumerate(algorithms):
			# Use the special queue that is intended for exchanging data among processes from the multiprocessing package.
			raw_cum = multiprocessing.Queue()
			raw_avg = multiprocessing.Queue()
			raw_psd = multiprocessing.Queue()
			
			cum_regret.append([0]*T)
			avg_regret.append([0]*T)
			psd_regret.append([0]*T)
			cum_regret_var.append([0]*T)
			avg_regret_var.append([0]*T)
			psd_regret_var.append([0]*T)
			start_time = time.perf_counter()
			processes = list()
			assigned_rpts = 0
			if rpts > AVAILABLE_CORES:
				required_processes = AVAILABLE_CORES
				rpts_per_process = math.ceil(rpts / AVAILABLE_CORES)
				if rpts_per_process != rpts / AVAILABLE_CORES:
					print("Warning: repetitions could not be evenly distributed.")
			else:
				required_processes = rpts
				rpts_per_process = 1
			for trial in range(required_processes):
				# Do the multiprocessing in the repetitions and block until all are done. Assumption: Running a fixed strategy on a fixed environment always takes roughly the same amount of time, so the blocking is no real issue.
				rpts_here = min(rpts_per_process, (rpts - assigned_rpts))
				process = multiprocessing.Process(target=run, args=(deepcopy(algorithm), deepcopy(env), raw_cum, raw_avg, raw_psd, rpts_here))
				process.start()
				print("Process spawned, performing", rpts_here, "repetitions.")
				processes.append(process)
				# If we could not evenly distribute, we might have assigned everything before running out of processes.
				assigned_rpts += rpts_here
				print(rpts-assigned_rpts, "left.")
				if assigned_rpts == rpts:
					break
			#end_time = time.perf_counter()
			#sum_times[i] += end_time - start_time
			
			# Problem: We have a list of repetitions with queues of turns, but we need a list of turns with lists of repetitions because we can only compute the variance AFTER we have the average of one turn in different repetitions. So swap everything to be arranged how we need it to avoid making things unnecessarily complicated.
			refined_cum = refine(raw_cum, T, rpts)
			refined_avg = refine(raw_avg, T, rpts)
			if hasattr(algorithm, "get_psd_rgt"):
				refined_psd = refine(raw_psd, T, rpts)
				has_pseudo = True
			
			# All data has been collected by now, so join the subprocesses.
			for process in processes:
				process.join()
			
			for y in range(T):
				cum_regret[-1][y], cum_regret_var[-1][y] = evaluate(refined_cum[y])
				avg_regret[-1][y], avg_regret_var[-1][y] = evaluate(refined_avg[y])
				if hasattr(algorithm, "get_psd_rgt"):
					psd_regret[-1][y], psd_regret_var[-1][y] = evaluate(refined_psd[y])
	
	#for i in range(len(algorithms)):
		#print("Average time for "+algorithm_names[i]+": "+str(sum_times[i] / (rpts*len(envs)))+" seconds.")
	
	plotOnce(env_names, algorithm_names, cum_regret, cum_regret_var, "Cumulative Regret", logscale)
	if has_pseudo:
		plotOnce(env_names, algorithm_names, psd_regret, psd_regret_var, "Pseudo Regret", logscale)
	plotOnce(env_names, algorithm_names, avg_regret, avg_regret_var, "Average Regret", logscale)
	
	plt.show()
