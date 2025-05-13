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

def compressSamples(samples, errors):
	global ERRORBAR_COUNT
	if ERRORBAR_COUNT == 0:
		return None, None, None
	if ERRORBAR_COUNT == 1:
		pos = round(len(samples) / 2)
		return [pos], [samples[pos]], [errors[pos]]
	if ERRORBAR_COUNT < 0:
		n = len(samples)
	else:
		n = min(ERRORBAR_COUNT, len(samples))
	compressed_range = [0]*n
	compressed_samples = [0]*n
	compressed_errors = [0]*n
	stepsize = (len(samples)-1) / (n-1) # minus one because our last sample is at len-1; again because we can have one more bar than steps
	for i in range(n):
		pos = round(stepsize*i)
		compressed_range[i] = pos
		compressed_samples[i] = samples[pos]
		compressed_errors[i] = errors[pos]
	return compressed_range, compressed_samples, compressed_errors

def readOneResult(filename):
	global results
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
	global results
	results = dict()
	
	if path == 0:
		if not "resultpath" in globals():
			resultpath = "results/"
		path = resultpath
	try:
		filenames = os.listdir(path)
	except:
		print("Folder", path, "does not exist (yet)")
		return
	if len(filenames) == 0:
		print("Folder", path, "is empty.")
	for filename in filenames:
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
	custom_colors = ['r','b','g','pink','orange']
	if len(data) > 5:
		auto_colors = list(plt.cm.jet(np.linspace(0, 1, len(data)-5)))
		all_colors = custom_colors + auto_colors
	else:
		all_colors = custom_colors
	colors = iter(all_colors)
	plt.figure(figsize=(6, 5))
	for what in data:
		color = next(colors)
		samples = data[what][0]
		samples_var = data[what][1]
		plt.plot(range(len(samples)), samples, label=what, color=color)
		#var_low, var_up = makeInterval(samples, samples_var)
		#plt.fill_between(range(len(samples)), var_low, var_up, color="xkcd:light grey")
		compressed_range, compressed_samples, compressed_errors = compressSamples(samples, samples_var)
		plt.errorbar(compressed_range, compressed_samples, yerr=compressed_errors, color=color, fmt="none")
		
	plt.xlabel('t (Trials)', fontsize=15)
	plt.ylabel(label, fontsize=15)
	plt.legend(loc='upper right')
	plt.title(label)
	if logscale:
		plt.xscale('log')


def plotResults(logscale=False):
	global ERRORBAR_COUNT
	if not "ERRORBAR_COUNT" in globals():
		ERRORBAR_COUNT = 40
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

def run(algorithm, env, raw_others, metric_names, time_passed, todo, refresh_first):
	for x in range(todo):
		start_time = time.perf_counter()
		if refresh_first:
			env.clear()
		algorithm.run(env)
		end_time = time.perf_counter()
		for name in metric_names:
			if name == "Cumulative Regret":
				raw_others[name].put(algorithm.get_cum_rgt())
			elif name == "Average Regret":
				raw_others[name].put(algorithm.get_avg_rgt())
			elif hasattr(algorithm, "listMetrics") and name in algorithm.listMetrics():
				raw_others[name].put(algorithm.getMetric(name))
		time_passed.put(end_time - start_time)
		
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
	


def test(T, rpts, envs, algorithms, algorithm_names, env_names, logscale=False, purge_old_results=True, refresh_first=False):
	if purge_old_results:
		purgeResults()
	testOnly(T, rpts, envs, algorithms, algorithm_names, env_names, refresh_first)
	readAllResults()
	
	# TODO: Handle this like the results (save and plot from file)
	for i in range(len(envs)):
		if hasattr(envs[i], "getMeans"):
			plotMeans(envs[i].getMeans(T), "Means of " + env_names[i])
	
	plotResults(logscale)


def testOnly(T, rpts, envs, algorithms, algorithm_names, env_names, refresh_first=False):
	# todo: move every non-standard metric here.
	metric_names = set(("Cumulative Regret", "Average Regret"))
	for algo in algorithms:
		if not hasattr(algo, "listMetrics"):
			continue
		for name in algo.listMetrics():
			metric_names.add(name)
	if "METRICS_WHITELIST" in globals():
		global METRICS_WHITELIST
		for metric in METRICS_WHITELIST:
			if not metric in metric_names:
				print("Could not exclude \""+metric+"\": It's not there.")
		outs = set()
		for metric in metric_names:
			if not metric in METRICS_WHITELIST:
				print("Excluding \""+metric+"\".")
				outs.add(metric)
		for metric in outs:
			metric_names.remove(metric)
	
	metrics = dict()
	metrics_var = dict()
	for name in metric_names:
		metrics[name] = []
		metrics_var[name] = []
	
	sum_times = [0]*len(algorithms)
	
	global AVAILABLE_CORES
	if not "AVAILABLE_CORES" in globals():
		AVAILABLE_CORES = 8 # How many subprocesses will be spawned at maximum. This number should not exceed the number of physical cores available on your system to avoid diminishing returns and crashes! If less repetitions are wanted, only that much processes will be spawned. If more repetitions are wanted, they will be equally distributed among the processes.
	
	global QUIET
	if not "QUIET" in globals():
		QUIET = False # Information on spawned processes will be output unless not desired.
	
	assert len(envs) == len(env_names) and len(algorithms) == len(algorithm_names)
	
	for env in envs:
		for i, algorithm in enumerate(algorithms):
			# Use the special queue that is intended for exchanging data among processes from the multiprocessing package.
			raw_others = dict()
			for name in metric_names:
				raw_others[name] = multiprocessing.Queue()
			time_passed = multiprocessing.Queue()
			
			for name in metric_names:
				metrics[name].append([0]*T)
				metrics_var[name].append([0]*T)
			
			processes = list()
			assigned_rpts = 0
			if rpts > AVAILABLE_CORES:
				required_processes = AVAILABLE_CORES
				rpts_per_process = math.ceil(rpts / AVAILABLE_CORES)
				if rpts_per_process != rpts / AVAILABLE_CORES and not QUIET:
					print("Warning: repetitions could not be evenly distributed.")
			else:
				required_processes = rpts
				rpts_per_process = 1
			for trial in range(required_processes):
				# Do the multiprocessing in the repetitions and block until all are done. Assumption: Running a fixed strategy on a fixed environment always takes roughly the same amount of time, so the blocking is no real issue.
				rpts_here = min(rpts_per_process, (rpts - assigned_rpts))
				process = multiprocessing.Process(target=run, args=(deepcopy(algorithm), deepcopy(env), raw_others, metric_names, time_passed, rpts_here, refresh_first))
				process.start()
				if not QUIET:
					print("Process spawned, performing", rpts_here, "repetitions.")
				processes.append(process)
				# If we could not evenly distribute, we might have assigned everything before running out of processes.
				assigned_rpts += rpts_here
				if not QUIET:
					print(rpts-assigned_rpts, "left.")
				if assigned_rpts == rpts:
					break
			
			# Problem: We have a list of repetitions with queues of turns, but we need a list of turns with lists of repetitions because we can only compute the variance AFTER we have the average of one turn in different repetitions. So swap everything to be arranged how we need it to avoid making things unnecessarily complicated.
			refined_others = dict()
			for name in metric_names:
				refined_others[name] = refine(raw_others[name], T, rpts)
			
			for x in range(rpts):
				sum_times[i] += time_passed.get()
			
			# All data has been collected by now, so join the subprocesses.
			for process in processes:
				process.join()
			
			for y in range(T):
				for name in metric_names:
					metrics[name][-1][y], metrics_var[name][-1][y] = evaluate(refined_others[name][y])
	
	for i in range(len(algorithms)):
		print("Average time for "+algorithm_names[i]+": "+str(sum_times[i] / (rpts*len(envs)))+" seconds.")
	
	for name in metric_names:
		#print(name)
		writeBatch(env_names, algorithm_names, metrics[name], metrics_var[name], name)
