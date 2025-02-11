import numpy as np
from scipy import optimize
from helpers.gini import gini
import math

class EnvMultiOutput:
	def __init__(self,num_arm,mu, noise, weights):
		self.num_arm = num_arm
		self.mu = mu
		self.noise = noise
		self.d = len(mu[0])
		self.weights = weights
		assert len(mu) == num_arm
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
		#print(result)
		#print("Optimal costs:", self.opt)
		
		# Just call this here because it needs to be called whenever mu has changed.
		self.calcPareto(mu)
	
	def getMeans(self, timesteps):
		means = [None]*self.num_arm
		for arm in range(self.num_arm):
			means[arm] = [None]*self.d
			for dimension in range(self.d):
				means[arm][dimension] = [self.mu[arm][dimension]]*timesteps
		return means
	
	def getMu(self):
		return self.mu
	
	def calcPareto(self, mu):
		# (re-)calculates the pareto front.
		self.pareto_front = set()
		for i in range(self.num_arm):
			fail = False
			for j in range(self.num_arm):
				if i == j:
					continue
				
				# If there is another element j that is at least as good as i in all dimensions and better than j in at least one dimension, then i is not in the pareto front.
				if(np.all(mu[j] <= mu[i]) and np.any(mu[j] < mu[i])):
					#print(i, "loses against", j)
					fail = True
					break
			
			if not fail:
				self.pareto_front.add(i)
		#print("Pareto front is", self.pareto_front)
		# Pareto front is ready.
		
		# Now, Pareto regret is defined as the minimum distance between THE MEAN of the chosen arm vs that of some arm in the pareto front. This means that the pareto regret is constant for a given arm as long as the means do not change and we can save much effort by just pre-computing the pareto regret for every arm.
		self.pareto_regrets = [0]*self.num_arm
		
		for arm in range(self.num_arm):
			if arm in self.pareto_front:
				# If the arm is in the pareto front, it is closest to itself and its pareto regret can be left as 0.
				continue
			
			# Now find the arm in the front with the minimum distance
			self.pareto_regrets[arm] = math.inf
			for front_arm in self.pareto_front:
				distance = math.dist(mu[arm], mu[front_arm])
				self.pareto_regrets[arm] = min(self.pareto_regrets[arm], distance)
		
		#print("Pareto regret values are", self.pareto_regrets)
	
	def getParetoRegret(self, arm):
		return self.pareto_regrets[arm]
