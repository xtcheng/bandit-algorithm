import numpy as np
from copy import deepcopy

def identity(x):
	return x

# This actually works the same as the normal environment, with the exception that the expectation values of the rewards, instead of being arbitrary, are calculated from a set of vectors that can be inquired by the agent plus a single vector that is unknown to the agent. The latter (associated with the user) is always fixed, the others (associated with the arms) may be renewed after each timestep.

class EnvContextual:
	def __init__(self,num_arm, theta, noise, renewing_arms, linkFunction=None):
		self.num_arm = num_arm
		self.theta = theta
		self.num_features = len(theta)
		self.renewing_arms = renewing_arms
		self.mu = [0]*self.num_arm
		self.arm_features = [None]*self.num_arm
		for i in range(self.num_arm):
			self.arm_features[i] = [0]*self.num_features
		self.noise = noise
		if linkFunction == None:
			# Unless we are in the general linear bandit setting and want to apply some possibly nonlinear link function to the dot product before considering it the expected reward, do nothing and pass through.
			self.linkFunction = identity
		else:
			# If a link function has been supplied, use that. You can pass any function as an argument (functions are more or less objects in python) that takes one number as argument and returns one number.
			self.linkFunction = linkFunction
		self.renewArms(force=True)
	
	def feedback(self,arm):
		rwd = self.mu[arm] + self.noise.sample_trunc()
		br  =  max(self.mu)
		# Ensure rewards are between 0 and 1.
		#rwd = max(0, min(1, rwd))
		#br = max(0, min(1, br))
		self.renewArms()
		return rwd, br
	
	def renewArms(self, force=False):
		if self.renewing_arms or force:
			for i in range(self.num_arm):
				self.arm_features[i] = np.random.uniform(0,1,self.num_features)
				self.mu[i] = self.linkFunction(np.dot(self.arm_features[i], self.theta))
	
	def getArmFeatures(self):
		return deepcopy(self.arm_features)
