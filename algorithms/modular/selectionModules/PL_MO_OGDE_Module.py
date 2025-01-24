import numpy as np
import math
from algorithms.modular.selectionModules.MO_OGDE_Module import MO_OGDE_Module

# Parameterized Learning MO OGDE
class PL_MO_OGDE_Module(MO_OGDE_Module):
	def __init__(self, history, T, num_arm, num_objectives, learning_factor, gini_weights):
		self.T = T
		self.num_arm = num_arm
		self.num_objectives = num_objectives
		self.learning_factor = learning_factor
		self.current_mix = np.array([1/num_arm]*num_arm)
		self.gini_weights = gini_weights
		self.history = history
		self.history.register(self)
		self.fullReset()
	
	def getLearningRate(self):
		return self.learning_factor * (math.sqrt(2/self.num_arm + 2*self.num_objectives**2) / math.sqrt(self.history.current_turn))
