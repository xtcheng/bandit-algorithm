import numpy as np
import sys
if not "../" in sys.path:
	sys.path.append('../')

class CW_OFUL_Module:
	def __init__(self,T,num_arm, num_features, lmda, beta, alpha):
		self.T = T
		self.num_arm = num_arm
		self.num_features = num_features
		self.lmda = lmda
		self.beta = beta
		self.alpha = alpha
		
		self.fullReset()
	
	def suggestArm(self):
		self.theta = np.matmul(np.linalg.inv(self.sigma) , self.b)
		p = np.zeros(self.num_arm)
		for arm in range(self.num_arm):
			p[arm] = np.dot(self.theta, self.arm_features[arm]) + self.beta * np.sqrt( np.matmul(self.arm_features[arm].T , np.matmul(np.linalg.inv(self.sigma) , self.arm_features[arm])) )
		return np.argmax(p)
	
	def thisHappened(self, arm, reward, timestep):
		self.sigma += self.weight * np.matmul(self.arm_features[arm].reshape(-1, 1), self.arm_features[arm].reshape(1, -1))
		self.b = self.b + self.weight * self.arm_features[arm] * reward
		self.weight = min(1, self.alpha / np.matmul(self.arm_features[arm].T , np.matmul(np.linalg.inv(self.sigma), self.arm_features[arm])))
	
	def knowArmFeatures(self, features):
		self.arm_features = features
	
	def fullReset(self):
		self.sigma = self.lmda * np.identity(self.num_features)
		self.b = np.zeros(self.num_features)
		self.weight = 1
