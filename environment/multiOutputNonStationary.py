import numpy as np
from scipy import optimize
from algorithms.modular.gini import gini

class EnvMultiOutputNonStationary:
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
		self.calcOptimalCosts()
		print("Optimal costs:", self.opt)
	
	def feedback(self,arm):
		if self.turns in self.breakpoints:
			self.position += 1
			# Things have changed, so recalculate the optimal gini index.
			self.calcOptimalCosts()
		rwd = np.zeros(self.d)
		for i in range(self.d):
			rwd[i] = self.mu[self.position][arm][i] + self.noise.sample_trunc()
		self.turns += 1
		return rwd, self.opt
	
	def calcOptimalCosts(self):
		ar = [self.weights]
		for m in self.mu[self.position]:
			ar.append(m)
		con = [{'type':'eq', 'fun': (lambda x : sum(x)-1)}]
		for i in range(3):
			con.append({'type':'ineq', 'fun': (lambda x : x[i])})
		self.opt = optimize.minimize(gini, args=ar, x0=[1/self.num_arm]*self.num_arm, constraints=con).fun
	
	def getMu(self):
		return self.mu[self.position]
