import numpy as np
from collections import deque

class MonitoredUCB:
	def __init__(self,T,num_arm, w, b, gamma):
		self.T = T
		self.num_arm = num_arm
		self.len_window = w
		self.detection_threshold = b
		self.gamma = gamma
		self.clear()
	
	
	def changeDetection(self, window, arm):
		# Assume that sum_rewards0 and sum_rewards already contain the correct results for the current point in time. This is true the first time this function is called for any arm (after start or an reset) because the number of pulls for that arm will equal len_window, so sum_rewards0 is the sum of rewards [1, len_window/2] and sum_rewards0 is the sum of [len_window/2 +1, len_window]. So just use the sums unmodified.
		result = np.abs(self.sums_rewards0[arm] - self.sums_rewards1[arm]) > self.detection_threshold
		#print("Checking arm", arm, ", value", np.abs(self.sums_rewards0[arm] - self.sums_rewards1[arm]), result)
		
		# Now make sure the invariant will also be true next time for this arm.
		
		# The oldest element in the left part of the current window will be outside the scope of the next window, so undo its addition and pop it so next time the 2nd-oldest one will be removed and so on.
		self.sums_rewards0[arm] -= window[0].popleft()
		
		# The oldest element in the right part of the current window will be inside the left part of the next window.
		temp = window[1].popleft()
		self.sums_rewards1[arm] -= temp
		self.sums_rewards0[arm] += temp
		window[0].append(temp)
		
		return result
	
	
	
	def run(self,env):
		
		# The variables used in this function only. As lists; not really need to use numpy-arrays here.
		num_play = [0]*self.num_arm
		sum_mu = [0]*self.num_arm
		ucb = [0]*self.num_arm
		
		# The moment that we consider to be the start of the observation.
		start = 0
		
		# Past rewards. List that for each arm contains a 2-tuple of deques of that arm's history.
		past_rewards = list()
		for i in range(self.num_arm):
			# Important: Use deque so that, unlike with list, not only append but also popleft runs in O(1) and inplace.
			past_rewards.append((deque(), deque()))
		
		########################################################
		
		for t in range(0,self.T):
			# Perform uniform exploration in the fraction of gamma.
			relative = (t-start) % int(self.num_arm / self.gamma)
			if relative < self.num_arm:
				arm = relative
			else:
				for j in range(self.num_arm):
					ucb[j] = sum_mu[j]/num_play[j] + np.sqrt(0.5*np.log(t+1-start)/num_play[j])
				arm = np.argmax(ucb)
			reward, optimal_reward = env.feedback(arm)
			
			# As before: Add the new reward. And increase the counter for the pulled arm.
			sum_mu[arm] += reward
			num_play[arm] += 1
			
			# Safe the rewards in the first deque until it is "full". Then add rewards only to the second list. They will eventually transist to the first list via the pop-logic in changeDetection.
			if num_play[arm] <= self.len_window/2:
				past_rewards[arm][0].append(reward)
				self.sums_rewards0[arm] += reward
			else:
				past_rewards[arm][1].append(reward)
				self.sums_rewards1[arm] += reward
			
			# Unless both parts of the window are not "full" yet, perform change detection.
			if num_play[arm] >= self.len_window:
				if self.changeDetection(past_rewards[arm], arm):
					print("Change detected at timestep", t)
					start = t+1 # Remember: We count from zero here.
					
					# Reset the helpers
					self.sums_rewards0 = [0]*self.num_arm
					self.sums_rewards1 = [0]*self.num_arm
					past_rewards = list()
					for i in range(self.num_arm):
						past_rewards.append((deque(), deque()))
					
					# Reset the past pulls. Also reset the sums because we do not keep all results and sum up everything anyway. mu can stay, it will be recalculation before its next use.
					num_play = [0]*self.num_arm
					sum_mu = [0]*self.num_arm
			
			
			# For performance analysis
			self.sum_rgt += (optimal_reward - reward)
			self.avg_rgt[t] += self.sum_rgt/(t+1)
			self.cum_rgt[t] += self.sum_rgt
	
	def get_avg_rgt(self):
		return self.avg_rgt
	
	def get_cum_rgt(self):
		return self.cum_rgt
	
	def clear(self):
		self.sum_rgt = 0
		self.avg_rgt = np.zeros(self.T)
		self.cum_rgt = np.zeros(self.T)
		
		# Needed for changeDetection: The sum of the left and the right part of the window, per arm.
		self.sums_rewards0 = [0]*self.num_arm
		self.sums_rewards1 = [0]*self.num_arm
