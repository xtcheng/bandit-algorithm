import numpy as np
import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.UCB1 import UCB1

class SlidingWindowUCB(UCB1):
	def __init__(self,T,num_arm, gamma):
		super().__init__(T, num_arm)
		# This time, gamma is how far the memory goes back into the past.
		self.gamma = gamma
		
		# This describes what arm was chosen at which time
		self.past_actions = [0]*gamma
		
		# And this describes what the reward of that chosen arm was.
		self.past_rewards = [0]*gamma
		
		# Use a C-style circular array. Actually, whith python's list having properties of linked lists, it should also be possible to simply use them as a FIFO without loosing much efficiency.
		self.pos = gamma-1
	
	def run(self,env):
		for i in range(0,self.T):
			self.pos = (self.pos + 1) % self.gamma
			if i < self.num_arm:
				arm = i
			else:
				arm = np.argmax(self.ucb)
			reward, br = env.feedback(arm)
			
			# New: Remove the impact of the oldest play-reward pair if memory is full. Attention: For the rewards, this may show a tiny deviation from summing up the whole array in every step because of floating point imprecision. The impact should be neglicible.
			if i >= self.gamma:
				out = self.past_actions[self.pos]
				self.sum_mu[out] -= self.past_rewards[out]
				self.num_play[out] -= 1
				# The number of plays is reduced by 1 instead of being multiplied with gamma for every arm in every step. The latter is how it is described in the paper, but that does not make any sene?
				
				# Now the mu of that arm has changed, so recalculate it if it is not the current arm, which will be refreshed anyway.
				if self.num_play[out] > 0:
					self.mu[out] = self.sum_mu[out]/self.num_play[out]
				else:
					self.mu[out] = np.inf
			
			# Save what happened this time.
			self.past_actions[self.pos] = arm
			self.past_rewards[self.pos] = reward
			
			# As before: Add the new reward. And increase the counter for the pulled arm.
			self.sum_mu[arm] += reward
			self.num_play[arm] += 1
			
			# Calculate new mu for the current arm.
			self.mu[arm] = self.sum_mu[arm]/self.num_play[arm]
			
			# As before.
			for j in range(self.num_arm):
				if(self.num_play[j]>0):
					self.ucb[j] = self.mu[j] + np.sqrt(2*np.log(min(i+1, self.gamma+1))/self.num_play[j])
				else:
					self.ucb[j] = np.inf
			
			self.sum_rgt += (br - reward)
			self.avg_rgt[i] += self.sum_rgt/(i+1)
			self.cum_rgt[i] += self.sum_rgt
