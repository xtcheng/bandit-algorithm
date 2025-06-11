import numpy as np
import sys
if not "../" in sys.path:
	sys.path.append('../')

class LinUCBModule:
	def __init__(self,T,num_arm, num_features, alpha):
		self.T = T
		self.num_arm = num_arm
		self.num_features = num_features
		self.alpha = alpha
		self.fullReset()
	
	def suggestArm(self):
		self.theta = np.matmul(np.linalg.inv(self.A) , self.b)
		p = np.zeros(self.num_arm)
		for arm in range(self.num_arm):
			p[arm] = np.dot(self.theta, self.arm_features[arm]) + self.alpha * np.sqrt( np.matmul(np.matmul(self.arm_features[arm].T , np.linalg.inv(self.A)) , self.arm_features[arm]) )
		"""max_v = 0
		max_i = 0
		for i in range(self.num_arm):
			if p[i][i] > max_v:
				max_v = p[i][i]
				max_i = i
		return max_i"""
		print(self.theta)
		return np.argmax(p)
	
	def thisHappened(self, arm, reward, timestep):
		if reward <= 0:
			print(reward)
		# reshape(-1, 1) means n subarrays with 1 entries each, but actually means 1xn matrix (alias normal vector) and not nx1.
		self.A = self.A + np.matmul(self.arm_features[arm].reshape(-1, 1), self.arm_features[arm].reshape(1, -1))
		self.b = self.b + self.arm_features[arm] * reward
	
	def knowArmFeatures(self, features):
		self.arm_features = features
	
	def fullReset(self):
		self.A = np.identity(self.num_features)
		self.b = np.zeros(self.num_features)
