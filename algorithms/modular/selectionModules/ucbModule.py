import numpy as np
import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.selectionModules.abstractSelectionModule import AbstractSelectionModule

class UCBModule(AbstractSelectionModule):
	def __init__(self,T,num_arm, xi):
		super().__init__(T, num_arm)
		self.xi = xi
	
	def suggestArm(self):
		ucb = [0]*self.num_arm
		for arm in range(self.num_arm):
			# Ensure each arm has been pulled at least once, then return the maximum index of the standard ucb calculation.
			if self.num_play[arm] > 0:
				mu = self.sum_mu[arm]/self.num_play[arm]
				ucb[arm] = mu + np.sqrt(self.xi*np.log(self.current_turn)/self.num_play[arm])
			else:
				return arm
		return np.argmax(ucb)
