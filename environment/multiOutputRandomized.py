import numpy as np
from scipy import optimize
from helpers.gini import gini
from environment.multiOutput import EnvMultiOutput
from environment.Gaussian_noise import Gaussian_noise

class EnvMultiOutputRandomized(EnvMultiOutput):
	def __init__(self, num_arm, num_objectives, weights):
		self.num_arm = num_arm
		self.d = num_objectives
		self.weights = weights
		self.clear()
	
	def clear(self):
		self.noise = Gaussian_noise(1,0, np.random.uniform(0, 1), None) # Important: The same noise for all entries of all arms, as opposed to the paper. Fully rework how noise/mu-distributions work in oder to fix.
		self.mu = [0]*self.num_arm
		for i in range(self.num_arm):
			self.mu[i] = [0]*self.d
			for j in range(self.d):
				self.mu[i][j] = np.random.uniform(0, 1)
		self.calcOptimalCosts(self.mu)
