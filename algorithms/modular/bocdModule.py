# Modified code from https://github.com/Ralami1859/Restarted-BOCPD
# For continuity reasons, I also modified some names. I usually write names of classes in uppercase and everything else in lowercase (which is the defacto standard for coding in most programming languages), and in this project, I used camelcase for names of classes and functions, and snakecase for parameters and variables.

import numpy as np
from algorithms.modular.bocdHelpers import *


# The algorithm originally is for detecting a change in a single arm. So to apply it to mulitple arms, we need one set of the variables per arm. The least confusing way to do this is to use another class for that.
class ArmContainer:
	def __init__(self):
		self.alphas = np.array([1])
		self.betas = np.array([1])
		self.forecaster_distribution = np.array([1])
		self.pseudo_dist = np.array([1])
		self.like1 = 1



class BOCDModule:
	def __init__(self, selection_module, num_arm, T):
		self.selection_module = selection_module
		self.num_arm = num_arm
		self.gamma = 1/T # Switching Rate
		self.containers = [0]*self.num_arm
		self.fullReset()
	
	
	def thisHappened(self, arm, reward, t):
		# Removed the loop. It is taken care of by the caller.
		
		# New: Perform the calculations on the set of variables for the current arm. With this method, the existing code can be applied with only minor modifications.
		current = self.containers[arm]
		
		estimated_best_expert = np.argmax(current.forecaster_distribution)
		# Restart precedure
		if not(estimated_best_expert == 0):
			# Reinitialization
			current.alphas = np.array([1])
			current.betas = np.array([1])
			current.forecaster_distribution = np.array([1])
			current.like1 = 1
			
			# New: also perform the reset on the part that selects an arm.
			self.selection_module.resetArm(arm)
			
			# Removed the list of last breakpoints. It was only used for output in the original.
		
		# Modified: This strategy expects the reward to come from a bernulli distribution, so take the actual reward and interpret it as the propability.
		bernulli_reward = int (np.random.uniform() < reward)
		(current.forecaster_distribution, current.pseudo_dist, current.like1) = updateForecasterDistribution_m(current.forecaster_distribution, current.pseudo_dist, current.alphas, current.betas, bernulli_reward, self.gamma, current.like1)
		(current.alphas, current.betas) = updateLaplacePrediction(current.alphas, current.betas, bernulli_reward) #Update the laplace predictor
	
	
	def fullReset(self):
		for i in range(self.num_arm):
			self.containers[i] = ArmContainer()
