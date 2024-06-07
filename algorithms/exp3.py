import numpy as np

class exp3:
	def __init__(self,T,num_arm, alpha, gamma):
		self.T = T
		self.K = num_arm
		self.alpha = alpha
		self.gamma = gamma
		self.rng = np.random.default_rng()
		self.clear()
		
	
	def run(self,env):
		for i in range(1,self.T+1): # important: Start from 1, because current time is used in the formula.
			
			# Step 1: Re-calculate the propabilities to pick an arm.
			for j in range(self.K):
				self.propabilities[j] = (1-self.gamma) * self.omegas[j] / self.total_omega + self.gamma/self.K
			#print(self.propabilities)
			
			# Step 2: Select random arm according to the distribution.
			arm_position = self.rng.random()
			arm = self.K - 1
			current_position = 0.0
			for j in range(self.K):
				current_position += self.propabilities[j]
				if current_position > arm_position:
					arm = j
					break
			#print("Random number", arm_position, ", pick arm", arm)
			
			# Step 3: Pull arm and receive result
			reward, best = env.feedback(arm)
			
			# Step 4: Update expectation for the chosen arm and omega for all.
			expectation = reward / self.propabilities[arm]
			self.total_omega = 0
			for j in range(self.K):
				inner_part = self.alpha / (self.propabilities[j] * np.sqrt(self.K*self.T))
				if j == arm:
					# x hat is 0 for the other arms, so imagine else: += 0
					inner_part += expectation
				self.omegas[j] = self.omegas[j] * np.exp((self.gamma/(3*self.K))*inner_part)
				self.total_omega += self.omegas[j]
			
			
			# For analysis only:
			self.sum_regret += (best - reward)
			self.avg_regret[i-1] += self.sum_regret/(i)
			self.cum_regret[i-1] += self.sum_regret
	
	def get_avg_rgt(self):
		return self.avg_regret
	
	def get_cum_rgt(self):
		return self.cum_regret
	
	def clear(self):
		# (Re-)init the omega values.
		self.omegas = np.zeros(self.K)
		init_value = np.exp((self.alpha*self.gamma/3) * np.sqrt(self.T / self.K))
		for i in range(self.K):
			self.omegas[i] = init_value
		self.total_omega = self.K * init_value
		
		# And the rest.
		self.propabilities = np.zeros(self.K)
		self.sum_regret = 0
		self.avg_regret = np.zeros(self.T)
		self.cum_regret = np.zeros(self.T)


"""
Note: The distribution is a bit tricky. Although it is easy to see
that the pi always sum up to 1, the most feasible way of implementing this
is interpret all pi as the size of an interval, with i being chosen if a
random number between 0 and 1 falls into this interval. One would have to
sum up the pi until the random number is reached, which means that the i whose
pi has been added last was chosen. However, rounding errors may occur, especially
with many number of greatly different magnitude.
"""
