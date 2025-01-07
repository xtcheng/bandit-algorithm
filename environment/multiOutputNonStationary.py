import numpy as np
from scipy import optimize
from helpers.gini import gini
from environment.multiOutput import EnvMultiOutput

class EnvMultiOutputNonStationary(EnvMultiOutput):
	def __init__(self,num_arm,mu, noise, weights, breakpoints):
		self.num_arm = num_arm
		self.mu = mu
		self.noise = noise
		self.d = len(mu[0][0])
		self.weights = weights
		self.breakpoints = breakpoints
		self.position = 0
		self.turns = 0
		assert len(self.breakpoints) == len(self.mu)-1 # are there really breakpoints+1 variations?
		for i in range(len(self.breakpoints)+1):
			assert len(mu[i]) == self.num_arm # Does each variation have the correct number of arms?
			for j in range(self.num_arm):
				assert len(mu[i][j]) == self.d # Does each arm return d values?
		self.calcOptimalCosts(self.mu[self.position])
	
	def feedback(self,arm):
		if self.turns in self.breakpoints:
			self.position += 1
			# Things have changed, so recalculate the optimal gini index.
			self.calcOptimalCosts(self.mu[self.position])
		rwd = np.zeros(self.d)
		for i in range(self.d):
			rwd[i] = self.mu[self.position][arm][i] + self.noise.sample_trunc()
		self.turns += 1
		return rwd, self.opt
	
	def getMu(self):
		return self.mu[self.position]
	
	def getMeans(self, timesteps):
		means = [None]*self.num_arm
		for arm in range(self.num_arm):
			means[arm] = [None]*self.d
			for dimension in range(self.d):
				means[arm][dimension] = [None]*timesteps
				current_state = 0
				for timestep in range(timesteps):
					if timestep in self.breakpoints:
						current_state += 1
					means[arm][dimension][timestep] = self.mu[current_state][arm][dimension]
		return means
	
	def clear(self):
		self.position = 0
		self.turns = 0
		self.calcOptimalCosts(self.mu[self.position])
