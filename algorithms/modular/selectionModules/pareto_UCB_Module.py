import numpy as np
import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.selectionModules.MO_OGDE_Module import MO_OGDE_Module

class Pareto_UCB_Module(MO_OGDE_Module):
	def __init__(self, history, T, num_arm, num_objectives, alpha, gini_weights):
		self.T = T
		self.num_arm = num_arm
		self.num_objectives = num_objectives
		self.alpha = alpha
		self.gini_weights = gini_weights
		self.history = history
		self.history.register(self)
		self.fullReset()
	
	def updateMix(self):
		ucb = [0]*self.num_arm
		for arm in range(self.num_arm):
			if self.history.num_play[arm] == 0:
				ucb[arm] = 0
			else:
				# We want to find arms with a small feedback, so substract the UCB-potential from the average instead of adding it. A high potential means the current exploration of the arm is poor and the costs might be lower (instead of higher!) than the average suggests.
				# Of course, it might as well be higher, but when selecting arms, we are only interested in the "it might be better than we think" part of "it might be different than we think".
				ucb[arm] = self.history.mu[arm] - np.sqrt(self.alpha*np.log(self.history.current_turn * (self.num_objectives * self.num_arm )**0.25 )) / self.history.num_play[arm]
		
		# calculate the pareto front
		self.current_mix = np.ones(self.num_arm)
		for j in range(self.num_arm):
			for i in range(self.num_arm):
				if i == j:
					continue
				# Again we are talking about costs, so an arm is not in the pareto front if it yields more than others, not less.
				if(np.all(ucb[j] <= ucb[i]) and np.any(ucb[j] < ucb[i])):
					#print(i, "loses against", j)
					self.current_mix[i] = 0
		self.current_mix /= sum(self.current_mix)
		#print(ucb)
		#print(self.current_mix)
		#print()
		
