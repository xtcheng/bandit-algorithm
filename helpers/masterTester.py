import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import numpy as np
import math
import multiprocessing
from copy import deepcopy
import os

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
	if type(average) == list:
		average = np.array(average)
	if type(variance) == list:
		variance = np.array(variance)
	return average-variance, average+variance


def readOneResult(filename):
	global results
	if not "results" in globals():
		results = dict()
	label, what = filename.split(".")[:2]
	if "/" in label:
		label = label.split("/")[-1]
	if not label in results:
		results[label] = dict()
	data = np.loadtxt(filename, delimiter=',')
	if type(data[0]) == np.ndarray:
		results[label][what] = data
	else:
		print("Warning:", filename, "seems to include no information on the standard deviation. Will plot without it.")
		results[label][what] = [data, [0]*len(data)]


# Read in the results in the standard resultfolder (or any other folder that contains results). This will prepare all the data there for plotting, so you do not need to re-run strategies that did not change to compare them against ones that did. Just make sure the folder does not contain outdated files from previous experiments, because it will to include ALL files that are in the folder.
def readAllResults(path=0):
	global resultpath
	if path == 0:
		if not "resultpath" in globals():
			resultpath = "results/"
		path = resultpath
	for filename in os.listdir(path):
		if len(filename) >= 4 and filename[-4:] == ".csv":
			readOneResult(path + filename)

def writeBatch(env_names, algorithm_names, samples, samples_var, label):
	global resultpath
	if not "resultpath" in globals():
		resultpath = "results/"
	os.makedirs(resultpath, exist_ok=True)
	for j, env in enumerate(env_names):
		for i, algorithm in enumerate(algorithm_names):
			filename = label + "."
			data = (samples[len(algorithm_names)*j + i], samples_var[len(algorithm_names)*j + i])
			filename += algorithm_names[i]+" on "+env_names[j] + ".csv"
			np.savetxt(resultpath + filename, data, delimiter=',')

def plotMeans(means, title):
	# means[arm][dimension][timestep]
	colors =  list(mcolors.TABLEAU_COLORS.items())
	T = range(len(means[0][0]))

	fig, axs = plt.subplots(len(means[0]), sharex=True, squeeze=True)
	
	for j in range(len(means[0])):
		for i in range(len(means)):
			label = "arm " + str(i+1)
			axs[j].plot(T,means[i][j], c=colors[i][1],label=label)
			axs[j].set_ylabel("dimension "+str(j+1), fontsize=7)
			plt.subplots_adjust(hspace = 0.05 * 1 )
	fig.suptitle(title, y=0.92)
	lines_labels = [axs[0].get_legend_handles_labels()]
	lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
	fig.subplots_adjust(bottom=0.2)
	plt.figlegend( lines, labels, loc = 'upper center', ncol=5, labelspacing=0.0 )
	plt.xlabel("t (Trials)")
	#plt.title(title)
	#plt.show()

def plotOnce(data, label, logscale):
	plt.figure(figsize=(6, 5))
	for what in data:
		samples = data[what][0]
		samples_var = data[what][1]
		plt.plot(range(len(samples)), samples, label=what)
		var_low, var_up = makeInterval(samples, samples_var)
		plt.fill_between(range(len(samples)), var_low, var_up, color="xkcd:light grey")
		
	plt.xlabel('t (Trials)', fontsize=15)
	plt.ylabel(label, fontsize=15)
	plt.legend(loc='upper right')
	plt.title(label)
	if logscale:
		plt.xscale('log')


def plotResults(logscale=False):
	global results
	for label in results:
		plotOnce(results[label], label, logscale)
	plt.show()

def refine(samples, timesteps, rpts):
	ret = [0]*timesteps
	for i in range(timesteps):
		ret[i] = list()
	
	for x in range(rpts):
		one_series = samples.get() # timesteps for ONE of the repetitions. get will block until the data is available.
		for i in range(timesteps):
			ret[i].append(float(one_series[i])) # cast to float to ensure it is not some weird numpy object that would cause trouble in the plotting.
	return ret

def run(algorithm, env, raw_cum, raw_avg, raw_eff, todo):
	for x in range(todo):
		algorithm.run(env)
		raw_cum.put(algorithm.get_cum_rgt())
		raw_avg.put(algorithm.get_avg_rgt())
		if hasattr(algorithm, "get_eff_rgt"):
			raw_eff.put(algorithm.get_eff_rgt())
		
		algorithm.clear()
		if hasattr(env, "clear"):
			env.clear()

def purgeResults():
	global resultpath
	if not "resultpath" in globals():
		resultpath = "results/"
	try:
		filenames = os.listdir(resultpath)
	except:
		print(resultpath, "is not available, so there are no files to purge.")
		return
	for filename in filenames:
		if len(filename) >= 4 and filename[-4:] == ".csv":
			os.remove(resultpath + filename)
	print(resultpath, "has been purged.")
	


def test(T, rpts, envs, algorithms, algorithm_names, env_names, logscale=False, purge_old_results=True):
	if purge_old_results:
		purgeResults()
	testOnly(T, rpts, envs, algorithms, algorithm_names, env_names)
	readAllResults()
	
	# TODO: Handle this like the results (save and plot from file)
	for i in range(len(envs)):
		if hasattr(envs[i], "getMeans"):
			plotMeans(envs[i].getMeans(T), "Means of " + env_names[i])
	
	plotResults(logscale)


def testOnly(T, rpts, envs, algorithms, algorithm_names, env_names):

	avg_regret = []
	avg_regret_var = []
	cum_regret = []
	cum_regret_var = []
	eff_regret = []
	eff_regret_var = []
	has_eff = False
	#sum_times = [0]*len(algorithms)
	
	AVAILABLE_CORES = 8 # How many subprocesses will be spawned at maximum. This number should not exceed the number of physical cores available on your system to avoid diminishing returns and crashes! If less repetitions are wanted, only that much processes will be spawned. If more repetitions are wanted, they will be equally distributed among the processes.
	
	assert len(envs) == len(env_names) and len(algorithms) == len(algorithm_names)
	
	for env in envs:
		for i, algorithm in enumerate(algorithms):
			# Use the special queue that is intended for exchanging data among processes from the multiprocessing package.
			raw_cum = multiprocessing.Queue()
			raw_avg = multiprocessing.Queue()
			raw_eff = multiprocessing.Queue()
			
			cum_regret.append([0]*T)
			avg_regret.append([0]*T)
			eff_regret.append([0]*T)
			cum_regret_var.append([0]*T)
			avg_regret_var.append([0]*T)
			eff_regret_var.append([0]*T)
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
				process = multiprocessing.Process(target=run, args=(deepcopy(algorithm), deepcopy(env), raw_cum, raw_avg, raw_eff, rpts_here))
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
			if hasattr(algorithm, "get_eff_rgt"):
				refined_eff = refine(raw_eff, T, rpts)
				has_eff = True
			
			# All data has been collected by now, so join the subprocesses.
			for process in processes:
				process.join()
			
			for y in range(T):
				cum_regret[-1][y], cum_regret_var[-1][y] = evaluate(refined_cum[y])
				avg_regret[-1][y], avg_regret_var[-1][y] = evaluate(refined_avg[y])
				if hasattr(algorithm, "get_eff_rgt"):
					eff_regret[-1][y], eff_regret_var[-1][y] = evaluate(refined_eff[y])
	
	#for i in range(len(algorithms)):
		#print("Average time for "+algorithm_names[i]+": "+str(sum_times[i] / (rpts*len(envs)))+" seconds.")
	
	
	writeBatch(env_names, algorithm_names, cum_regret, cum_regret_var, "Cumulative Regret")
	if has_eff:
		writeBatch(env_names, algorithm_names, eff_regret, eff_regret_var, "Effective Regret")
	writeBatch(env_names, algorithm_names, avg_regret, avg_regret_var, "Average Regret")
