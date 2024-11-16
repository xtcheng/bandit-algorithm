import numpy as np

class DiscountModule:
	def __init__(self, selection_module, num_arm, gamma):
		self.selection_module = selection_module
		self.num_arm = num_arm
		assert gamma <=1 and gamma >=0
		self.gamma = gamma
		
		self.fullReset()
	
	
	
	def thisHappened(self, arm, reward, t):
		self.selection_module.current_turn = 0
		for j in range(self.num_arm):
			# Discount the past. This way, one gamma is multiplied per step into the past, without needing to save all of them.
			self.selection_module.sum_mu[j] *= self.gamma
			self.selection_module.num_play[j] *= self.gamma
			
			# Keep track of what the sum of all discounted plays is.
			self.selection_module.current_turn += self.selection_module.num_play[j]
			
			# Now the mu of that arm has changed, so recalculate it
			if self.selection_module.num_play[j] > 0:
				self.selection_module.mu[j] = self.selection_module.sum_mu[j]/self.selection_module.num_play[j]
		
	
	
	def fullReset(self):
		pass
	
