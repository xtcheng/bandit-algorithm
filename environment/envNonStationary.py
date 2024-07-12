
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:30:21 2023
@author: Xiaotong
"""

import numpy as np

class env_non_stationary:
	def __init__(self,num_arm,mu, noise, breakpoints):
		self.num_arm = num_arm
		self.mu = mu
		self.noise = noise
		self.breakpoints = breakpoints
		self.position = 0
		self.turns = 0
		self.rng = np.random.default_rng()
	
	def feedback(self,arm):
		if self.turns in self.breakpoints:
			self.position += 1
		rwd = self.mu[self.position][arm] + self.noise.sample_trunc()
		br  =  max(self.mu[self.position])
		self.turns += 1
		return rwd, br
	
	def clear(self):
		self.position = 0
		self.turns = 0
	
	"""def feedbackMult(self,arms):
		if self.turns in self.breakpoints:
			self.position
		# Draw the selected arms and output 0 for all others.
		rewards = list()
		for i in range(self.num_arm):
			if i in arms:
				rewards.append(self.mu[self.position][i] + self.noise.sample_trunc())
			else:
				rewards.append(0)
		# The best possible reward is by pulling the k best arms.
		br = 0
		for expectation in sorted(self.mu, reverse=True)[:len(arms)]:
			br += expectation
		self.shuffleMaybe()
		return rewards, br"""
