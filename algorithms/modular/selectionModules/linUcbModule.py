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
		theta = np.linalg.inv(self.A) * self.b
		p = [0]*self.num_arm
		for arm in range(self.num_arm):
			print(theta)
			print(self.arm_features[arm])
			print(np.transpose(theta)*self.arm_features[arm])
			#problem: I multiply a dxd matrix with a dx1 matrix, so the result should be a dx1 matrix, but it is a dxd matrix instead.
			p[arm] = np.transpose(theta)*self.arm_features[arm] + self.alpha * np.sqrt( np.transpose(self.arm_features[arm]) * np.linalg.inv(self.A) * self.arm_features[arm] )
		max_v = 0
		max_i = 0
		for i in range(self.num_arm):
			if p[i][i] > max_v:
				max_v = p[i][i]
				max_i = i
		return max_i
	
	def thisHappened(self, arm, reward, timestep):
		self.A = self.A + self.arm_features[arm] * np.transpose(self.arm_features[arm])
		self.b = self.b + self.arm_features[arm] * reward
	
	def knowArmFeatures(self, features):
		self.arm_features = features
	
	def fullReset(self):
		self.A = np.identity(self.num_features)
		self.b = np.zeros(self.num_features)
