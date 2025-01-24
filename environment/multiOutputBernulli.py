import numpy as np
from scipy import optimize
from helpers.gini import gini
from environment.multiOutput import EnvMultiOutput
import random

class EnvMultiOutputBernulli(EnvMultiOutput):
	def __init__(self,num_arm, mu, weights):
		self.num_arm = num_arm
		self.mu = mu
		# no noise here because we interpret the mu as bernulli probabilities.
		self.d = len(mu[0])
		self.weights = weights
		for i in range(1, num_arm):
			assert len(mu[i-1]) == len(mu[i])
		self.calcOptimalCosts(self.mu)
		print("Optimal costs:", self.opt)
	
	def feedback(self,arm):
		rwd = np.zeros(self.d)
		for i in range(self.d):
			if random.random() < self.mu[arm][i]: # random covers [0,1), so for mu=1 this will always be true and for 0 always false.
				rwd[i] = 1
			else:
				rwd[i] = 0
		return rwd, self.opt
