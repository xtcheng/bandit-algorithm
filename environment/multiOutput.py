import numpy as np
from scipy import optimize
from algorithms.modular.gini import gini

class EnvMultiOutput:
	def __init__(self,num_arm,mu, noise, weights):
		self.num_arm = num_arm
		self.mu = mu
		self.noise = noise
		self.d = len(mu[0])
		self.weights = weights
		for i in range(1, num_arm):
			assert len(mu[i-1]) == len(mu[i])
		self.calcOptimalCosts()
		print("Optimal costs:", self.opt)
	
	def feedback(self,arm):
		rwd = np.zeros(self.d)
		for i in range(self.d):
			rwd[i] = self.mu[arm][i] + self.noise.sample_trunc()
		return rwd, self.opt
	
	def calcOptimalCosts(self):
		ar = [self.weights]
		for m in self.mu:
			ar.append(m)
		con = [{'type':'eq', 'fun': (lambda x : sum(x)-1)}]
		for i in range(3):
			con.append({'type':'ineq', 'fun': (lambda x : x[i])})
		self.opt = optimize.minimize(gini, args=ar, x0=[1/self.num_arm]*self.num_arm, constraints=con).fun
	
	def getMu(self):
		return self.mu
