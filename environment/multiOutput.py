import numpy as np

class EnvMultiOutput:
	def __init__(self,num_arm,mu, noise):
		self.num_arm = num_arm
		self.mu = mu
		self.noise = noise
		self.d = len(mu[0])
		for i in range(1, num_arm):
			assert len(mu[i-1]) == len(mu[i])
	
	def feedback(self,arm):
		rwd = np.zeros(self.d)
		for i in range(self.d):
			rwd[i] = self.mu[arm][i] + self.noise.sample_trunc()
		br = 42 # TODO
		return rwd, br
