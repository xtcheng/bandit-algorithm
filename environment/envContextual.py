from environment.env import env_stochastic
import numpy as np

# This actually works the same as the normal environment, with the exception that the expectation values of the rewards, instead of being arbitrary, are calculated from a set of vectors that can be inquired by the agent plus a single vector that is unknown to the agent. The latter (associated with the user) is always fixed, the others (associated with the arms) may be renewed after each timestep.

class EnvContextual(env_stochastic):
	def __init__(self,num_arm, user_features, noise, renewing_arms):
		self.num_arm = num_arm
		self.user_features = user_features
		self.num_features = len(user_features)
		self.renewing_arms = renewing_arms
		self.mu = [0]*self.num_arm
		self.arm_features = [None]*self.num_arm
		for i in range(self.num_arm):
			self.arm_features[i] = [0]*self.num_features
		self.noise = noise
		self.stability = 1
		self.rng = np.random.default_rng()
		self.renewArms()
	
	def feedback(self,arm):
		ret = super().feedback(arm)
		# Ensure rewards are between 0 and 1.
		ret = (max(0, min(1, ret[0])) , max(0, min(1, ret[1])))
		if self.renewing_arms:
			self.renewArms()
		return ret
	
	def renewArms(self):
		for i in range(self.num_arm):
			self.mu[i] = 0
			for j in range(self.num_features):
				self.arm_features[i][j] = self.rng.uniform()
				# Also calculate the new reward expectations in one go.
				self.mu[i] += self.arm_features[i][j] * self.user_features[j]
	
	def getArmFeatures(self):
		return self.arm_features
