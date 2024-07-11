import numpy as np
import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.UCB1 import UCB1

class DiscountedUCB(UCB1):
	def __init__(self,T,num_arm, gamma):
		super().__init__(T, num_arm)
		self.gamma = gamma
	
	def run(self,env):
		for i in range(0,self.T):
			if i < self.num_arm:
				arm = i
			else:
				arm = np.argmax(self.ucb)
			reward, br = env.feedback(arm)
			
			
			# New: Lessen the impact of past data. Note that in both the formula for the number of plays per arm and for the discounted empirical average, for each step into the past from the current time, the entry is multiplied with an additional gamma and the current entry enters the sum as is. Therefore, instead of actually applying the formula, it is sufficient to multiply the complete sum of the previous timestep with gamma, then add the new entry.
			total_plays = 0
			for j in range(self.num_arm):
				self.sum_mu[j] *= self.gamma
				# But the current arm is increased again.
				if j == arm:
					self.sum_mu[j] += reward
				
				self.num_play[j] *= self.gamma
				# Here too..
				if j == arm:
					self.num_play[j] += 1
				
				# Keep track of what the sum of all discounted plays is.
				total_plays += self.num_play[j]
				
				# Refresh mu.
				if self.num_play[j] > 0:
					self.mu[j] = self.sum_mu[j]/self.num_play[j]
				else:
					self.mu[j] = np.inf
			
			# Same as before.
			for j in range(self.num_arm):
				if(self.num_play[j]>0):
					self.ucb[j] = self.mu[j] + np.sqrt(2*np.log(total_plays)/self.num_play[j])
				else:
					self.ucb[j] = np.inf
			
			self.sum_rgt += (br - reward)
			self.avg_rgt[i] += self.sum_rgt/(i+1)
			self.cum_rgt[i] += self.sum_rgt
