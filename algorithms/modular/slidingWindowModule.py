import numpy as np

class SlidingWindowModule:
	def __init__(self, selection_module, num_arm, gamma):
		self.selection_module = selection_module
		self.num_arm = num_arm
		self.gamma = gamma
		
		# This describes what arm was chosen at which time
		self.past_actions = [0]*gamma
		
		# And this describes what the reward of that chosen arm was.
		self.past_rewards = [0]*gamma
		
		# Use a C-style circular array. It would also be possible to simply use a deque without loosing much efficiency.
		self.pos = gamma-1
		
		self.current_turn = 0
		
		self.fullReset()
	
	
	
	def thisHappened(self, arm, reward):
		# Remove the impact of the oldest play-reward pair if memory is full. Attention: For the rewards, this may show a tiny deviation from summing up the whole array in every step because of floating point imprecision. The impact should be neglicible.
		# Attention: It is removed AFTER the current turn, so use >=. For example, with window size 10, the refresh happens after turn 10, reducing the elements to 9, but then 1 more is inserted, and removed after evaluation of turn 11 has finished, indeed making the effective window size 10.
		
		self.current_turn += 1
		self.pos = (self.pos+1) % self.gamma
		self.past_rewards[self.pos] = reward
		self.past_actions[self.pos] = arm
		
		if self.current_turn >= self.gamma:
			out = self.past_actions[self.pos]
			self.selection_module.sum_mu[out] -= self.past_rewards[self.pos]
			self.selection_module.num_play[out] -= 1
			
			# Now the mu of that arm has changed, so recalculate it if it
			if self.selection_module.num_play[out] > 0:
				self.selection_module.mu[out] = self.selection_module.sum_mu[out]/self.selection_module.num_play[out]
			else:
				self.selection_module.mu[out] = np.inf
	
	
	def fullReset(self):
		self.current_turn = 0
	
