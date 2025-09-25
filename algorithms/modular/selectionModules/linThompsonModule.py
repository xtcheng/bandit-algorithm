import numpy as np
import sys
if not "../" in sys.path:
	sys.path.append('../')

# refer to https://github.com/yang0110/Bandit-algorithms/blob/master/lints.py
class LinThompsonModule:
	def __init__(self,T,num_arm, num_features, alpha, delta, sigma):
		self.T = T
		self.num_arm = num_arm
		self.num_features = num_features
		self.alpha = alpha
		self.delta = delta # prob to be wrong
		self.sigma = sigma # gauß noise
		self.v = self.sigma*np.sqrt(24/0.5*self.num_features*np.log(1/self.delta))
		self.fullReset()
	
	def suggestArm(self):
		est_y_list=np.zeros(self.num_arm)
		cov_inv=np.linalg.pinv(self.cov)
		sample_theta=np.random.multivariate_normal(mean=self.theta, cov=self.v*cov_inv)
		for i in range(self.num_arm):
			x=self.arm_features[i]
			est_y_list[i]=np.dot(sample_theta, x)
		
		index=np.argmax(est_y_list)
		return index
	
	def thisHappened(self, arm, reward, timestep):
		x = self.arm_features[arm]
		self.cov+=np.outer(x,x)
		self.bias+=x*reward
		self.theta=np.dot(np.linalg.pinv(self.cov), self.bias)
	
	def knowArmFeatures(self, features):
		self.arm_features = features
	
	def fullReset(self):
		self.cov=self.alpha*np.identity(self.num_features)
		self.bias=np.zeros(self.num_features)
		self.theta=np.zeros(self.num_features)
