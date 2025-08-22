import numpy as np
import sys
from scipy import optimize
if not "../" in sys.path:
	sys.path.append('../')
from helpers.commonTools import unit, vectorLen




class GeneralUCBModule:
	def __init__(self,T,num_arm, num_features, alpha, link_function):
		self.T = T
		self.num_arm = num_arm
		self.num_features = num_features
		self.alpha = alpha
		self.link_function = link_function
		self.optimization_steps = 1000
		self.rng = np.random.default_rng()
		self.fullReset()
	
	def suggestArm(self):
		# "If fprime is a boolean and is True, f is assumed to return the value of the objective function and of the derivative. fprime can also be a callable returning the derivative of f. In this case, it must accept the same arguments as f."
		#sol = optimize.minimize(self.targetFunction, x0=self.theta, method='powell', bounds=[(0,1)]*self.num_features)
		#sol = optimize.root(self.targetFunction, x0=self.theta)
		#print(sol)
		#self.theta = sol.x
		
		if self.current_time == 0:
			self.theta = np.zeros(self.num_features)
		else:
			self.theta = self.bgdMinimizer()
		
		p = np.zeros(self.num_arm)
		for arm in range(self.num_arm):
			p[arm] = self.link_function(np.dot(self.theta, self.arm_features[arm])) + self.alpha * np.sqrt( np.matmul(np.matmul(self.arm_features[arm].T , np.linalg.inv(self.A)) , self.arm_features[arm]) )
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
		result = np.zeros(len(theta_test))
		for k in range(self.current_time):
			result += (self.past_rewards[k] - self.link_function(np.dot(self.past_arms[k], theta_test))) * self.past_arms[k]
		#print("Result:", result)
		return result
	
	def targetFunctionMinimize(self, theta_test):
		result = 0
		for k in range(self.current_time):
			# Simply the cumulative error.
			result += self.past_rewards[k] - self.link_function(np.dot(self.past_arms[k], theta_test))
		return np.abs(result)
	
	def bgdMinimizer(self):
		# Reuse the online minimizer from the bgd continous optimization.
		
		# We need the output range. Estimate it by using the number of past turns. Might multiply some factor on top of it.
		C = self.current_time
		
		input_offsets = np.array([0.5]*self.num_features)
		y = self.theta
		R = np.sqrt((0.5**2) * self.num_features)
		r = 0.5
		
		# Calculate the parameters for the general case.
		ny = R / (C*np.sqrt(self.optimization_steps))
		delta = np.power((r * (R**2) * (self.num_features**2)) / (12*self.optimization_steps), 1/3)
		alpha = np.power((3*R*self.num_features) / (2*r*np.sqrt(self.optimization_steps)), 1/3)
		
		for timestep in range(1, self.optimization_steps+1):
			u = unit(self.rng.normal(size=self.num_features))
			x = y + delta*u
			costs = self.targetFunctionMinimize(x + input_offsets)
			if costs < self.smallest_yet:
				self.result = (x + input_offsets)
				self.smallest_yet = costs
			
			y_raw = y - ny*costs*u
			
			# And project it into the sphere
			if vectorLen(y_raw) > (1-alpha) * R:
				y = (1-alpha) * R * unit(y_raw)
			else:
				y = y_raw
		print("Result", self.smallest_yet, "with input", self.result)
		self.smallest_yet += 0.5
		return self.result
	
	def fullReset(self):
		self.A = np.identity(self.num_features)
		self.past_rewards = [None]*self.T
		self.past_arms = [None]*self.T
		self.current_time = 0
		self.theta = np.zeros(self.num_features)
		
		self.result = np.zeros(self.num_features)
		self.smallest_yet = np.inf
