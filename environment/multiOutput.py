import numpy as np
from scipy import optimize
from helpers.gini import gini

class EnvMultiOutput:
	def __init__(self,num_arm,mu, noise, weights):
		self.num_arm = num_arm
		self.mu = mu
		self.noise = noise
		self.d = len(mu[0])
		self.weights = weights
		for i in range(1, num_arm):
			assert len(mu[i-1]) == len(mu[i])
		self.calcOptimalCosts(self.mu)
		print("Optimal costs:", self.opt)
	
	def feedback(self,arm):
		rwd = np.zeros(self.d)
		for i in range(self.d):
			rwd[i] = self.mu[arm][i] + self.noise.sample_trunc()
		return rwd, self.opt
	
	def calcOptimalCosts(self, mu):
		ar = [self.weights]
		for m in mu:
			ar.append(m)
		con = [{'type':'eq', 'fun': (lambda x : sum(x)-1)}]
		result = optimize.minimize(gini, args=ar, x0=[1/self.num_arm]*self.num_arm, constraints=con, bounds=[(0,1)]*self.num_arm)
		self.opt = result.fun
		print(result)
		#print("Optimal costs:", self.opt)
	
	def getMu(self):
		return self.mu
