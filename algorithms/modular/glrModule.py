import numpy as np
from scipy.special import zeta, lambertw

class GLRModule:
	def __init__(self, selection_module, num_arm, delta, globalRestart):
		self.selection_module = selection_module
		self.num_arm = num_arm
		self.delta = delta
		self.globalRestart = globalRestart
		self.fullReset()
	
	
	def klFunction(self, x, y):
		#print(x, y, np.log((1-x) / (1-y)))
		return x*np.log(x/y) + (1-x)*np.log((1-x) / (1-y))
	
	def betaFunction(self, n, delta):
		return 2*self.tFunction(np.log(3*n*np.sqrt(n)/delta)/2) + 6*np.log(1 + np.log(n))
	
	
	def tFunction(self, x):
		return 2*self.hTildeFunction((self.hInverseFunction(1+x) + np.log(2*zeta(2))) / 2)
	
	
	def hInverseFunction(self, u):
		# h(u) = u - ln(u). Its inverse, according to calculator, is -W(-e^(-u)).
		return -lambertw(-(np.e**(-u)))
	
	
	def hTildeFunction(self, x):
		assert x >= 0
		if x > self.hInverseFunction(1 / np.log(3/2)):
			return np.e**(1/self.hInverseFunction(x)) * self.hInverseFunction(x)
		else:
			return (3/2)*(x - np.log(np.log(3/2)))
	
	
	def thisHappened(self, arm, reward):
		# Append the reward to the list of rewards for the pulled arm.
		self.past_rewards[arm].append(reward)
		self.past_rewards_sums[arm] += reward
		
		n=len(self.past_rewards[arm])
		if n > 2:
			# Perform the GLR-check. If any s is found that satisfies the formula, a restart has to be performed.
			mean = self.past_rewards_sums[arm] / len(self.past_rewards[arm])
			left_mean = 0
			right_mean = mean
			supreme = -1
			for s in range(1, n-1):
				z = self.past_rewards[arm][s-1]
				left_mean = ((s-1)*left_mean + z) / s
				right_mean = ((n+2-s)*right_mean) / (n-s+2)
				
				# Now calculate according to the formula.
				#print("arm=",arm,", s=",s,", n=",n, sep=""),
				#challenge = s * self.klFunction(sum_to_s / s, mean) + (n - s) * self.klFunction(sum_from_s / (n - s), mean)
				challenge = s * self.klFunction(left_mean, mean) + (n - s) * self.klFunction(right_mean, mean)
				
				# Is this the "supreme" challenge so far?
				if challenge > supreme:
					supreme = challenge
			
			# Now compare this to the confidency function.
			if supreme >= self.betaFunction(len(self.past_rewards[arm]), self.delta):
				print("Detected change in arm", arm, "after", len(self.past_rewards[arm]), "pulls")
				print("supreme was", supreme, "and result of beta was", self.betaFunction(len(self.past_rewards[arm]), self.delta))
				self.restart(arm)
	
	
	def fullReset(self):
		# This strategy has to keep track of all rewards an arm has yielded since start or reset and access all of them. So use standard list or numpy vector.
		self.past_rewards = list()
		for i in range(self.num_arm):
			self.past_rewards.append(list())
		
		# Also keep track of the current sum of the rewards to avoid a few calculations.
		self.past_rewards_sums = [0]*self.num_arm
	
	
	def restart(self, arm):
		if self.globalRestart:
			self.selection_module.fullReset()
			self.fullReset()
		else:
			self.selection_module.resetArm(arm)
			self.past_rewards[arm] = list()
			self.past_rewards_sums[arm] = 0
