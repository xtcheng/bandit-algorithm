import numpy as np
import sys
from scipy import optimize
if not "../" in sys.path:
	sys.path.append('../')




class GeneralUCBModule:
	def __init__(self,T,num_arm, num_features, alpha, link_function):
		self.T = T
		self.num_arm = num_arm
		self.num_features = num_features
		self.alpha = alpha
		self.link_function = link_function
		self.fullReset()
	
	def suggestArm(self):
		# "If fprime is a boolean and is True, f is assumed to return the value of the objective function and of the derivative. fprime can also be a callable returning the derivative of f. In this case, it must accept the same arguments as f."
		sol = optimize.minimize(self.targetFunction, x0=self.theta, method='powell', bounds=[(0,1)]*self.num_features)
		#print(sol)
		self.theta = sol.x
		
		p = np.zeros(self.num_arm)
		for arm in range(self.num_arm):
			p[arm] = np.dot(self.theta, self.arm_features[arm]) + self.alpha * np.sqrt( np.matmul(np.matmul(self.arm_features[arm].T , np.linalg.inv(self.A)) , self.arm_features[arm]) )
		return np.argmax(p)
	
	def thisHappened(self, arm, reward, timestep):
		# reshape(-1, 1) means n subarrays with 1 entries each, but actually means 1xn matrix (alias normal vector) and not nx1.
		self.A = self.A + np.matmul(self.arm_features[arm].reshape(-1, 1), self.arm_features[arm].reshape(1, -1))
		#self.b = self.b + self.arm_features[arm] * reward
		self.past_rewards[timestep] = reward
		self.past_arms[timestep] = self.arm_features[arm]
		self.current_time += 1
	
	def knowArmFeatures(self, features):
		self.arm_features = features
	
	def targetFunction(self, theta_test):
		#print(theta_test)
		result = 0
		for k in range(self.current_time):
			result += 100 * (self.past_rewards[k] - self.link_function(np.dot(self.past_arms[k], theta_test)))
		#print("Result:", result)
		return np.abs(result)
	
	def fullReset(self):
		self.A = np.identity(self.num_features)
		self.past_rewards = [None]*self.T
		self.past_arms = [None]*self.T
		self.current_time = 0
		self.theta = np.zeros(self.num_features)
