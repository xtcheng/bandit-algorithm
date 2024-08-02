import numpy as np
import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.abstractSelectionModule import AbstractSelectionModule

# Like ucb, but with forced exploration.

class UCBForcedExploreModule(AbstractSelectionModule):
	def __init__(self,T,num_arm, xi, alpha):
		super().__init__(T, num_arm)
		self.xi = xi
		self.alpha = alpha
	
	def suggestArm(self):
		if self.alpha > 0 and self.current_turn % np.floor(self.num_arm/self.alpha) < self.num_arm:
			return int(self.current_turn % np.floor(self.num_arm/self.alpha))
		ucb = [0]*self.num_arm
		for arm in range(self.num_arm):
			# Ensure each arm has been pulled at least once, then return the maximum index of the standard ucb calculation.
			if self.num_play[arm] > 0:
				mu = self.sum_mu[arm]/self.num_play[arm]
				ucb[arm] = mu + np.sqrt(self.xi*np.log(self.current_turn)/self.num_play[arm])
			else:
				return arm
		return np.argmax(ucb)
