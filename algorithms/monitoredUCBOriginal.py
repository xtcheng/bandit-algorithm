# This is the same code as in the other file, except for the changeDetection-function. Here, it is actually implemented as instructed by the paper - which is more inefficient compared to the other implementation.

import numpy as np
from collections import deque

class MonitoredUCBOriginal:
	def __init__(self,T,num_arm, w, b, gamma):
		self.T = T
		self.num_arm = num_arm
		self.len_window = w
		self.detection_threshold = b
		self.gamma = gamma
		self.clear()
	
	
	def changeDetection(self, window, arm):
		# Sum up through ALL entries in both parts of the window and compare them. This is simpler than the other solution, but less efficient.
		sum1 = 0
		sum2 = 0
		for i in range(0, int(self.len_window/2)):
			sum1 += window[(i + self.current_indexes[arm]) % self.len_window]
		for i in range(int(self.len_window/2), self.len_window):
			sum2 += window[(i + self.current_indexes[arm]) % self.len_window]
		return np.abs(sum1 - sum2) > self.detection_threshold
	
	
	
	def run(self,env):
		
		# The variables used in this function only. As lists; not really need to use numpy-arrays here.
		num_play = [0]*self.num_arm
		sum_mu = [0]*self.num_arm
		ucb = [0]*self.num_arm
		
		# The moment that we consider to be the start of the observation.
		start = 0
		
		# Past rewards. List that for each arm contains a list that arm's history.
		past_rewards = list()
		for i in range(self.num_arm):
			past_rewards.append([0]*self.len_window)
		
		########################################################
		
		for t in range(0,self.T):
			# Perform uniform exploration in the fraction of gamma.
			relative = (t-start) % int(self.num_arm / self.gamma)
			if relative < self.num_arm:
				arm = relative
			else:
				for j in range(self.num_arm):
					ucb[j] = sum_mu[j]/num_play[j] + np.sqrt(0.5*np.log(t+1-start)/num_play[j])
				arm = np.argmax(ucb)
			reward, optimal_reward = env.feedback(arm)
			
			# As before: Add the new reward. And increase the counter for the pulled arm.
			sum_mu[arm] += reward
			num_play[arm] += 1
			
			# Safe the rewards in the according list.
			past_rewards[arm][self.current_indexes[arm]] = reward
			self.current_indexes[arm] = (self.current_indexes[arm] + 1) % self.len_window
			
			# Unless the window is not "full" yet, perform change detection.
			if num_play[arm] >= self.len_window:
				if self.changeDetection(past_rewards[arm], arm):
					print("Change detected at timestep", t)
					start = t+1 # Remember: We count from zero here.
					
					# No need to reset the circular lists - they won't be used until full anyway.
					
					# Reset the past pulls. Also reset the sums because we do not keep all results and sum up everything anyway. mu can stay, it will be recalculation before its next use.
					num_play = [0]*self.num_arm
					sum_mu = [0]*self.num_arm
			
			
			# For performance analysis
			self.sum_rgt += (optimal_reward - reward)
			self.avg_rgt[t] += self.sum_rgt/(t+1)
			self.cum_rgt[t] += self.sum_rgt
	
	def get_avg_rgt(self):
		return self.avg_rgt
	
	def get_cum_rgt(self):
		return self.cum_rgt
	
	def clear(self):
		self.sum_rgt = 0
		self.avg_rgt = np.zeros(self.T)
		self.cum_rgt = np.zeros(self.T)
		
		# To keep track of where we are in the circular lists.
		self.current_indexes = [0]*self.num_arm
