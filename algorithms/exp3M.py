import numpy as np

class exp3:
	def __init__(self,T,num_arm, gamma):
		self.T = T
		self.K = num_arm
		self.gamma = gamma
		self.rng = np.random.default_rng()
		self.clear()
		
	
	def calcResult(alpha):
		# Calculate the left part of the equation to be satisfied using a given alpha.
		lower = 0
		for omega in self.omegas:
			if omega >= alpha:
				lower += alpha
			else:
				lower += omega
		return alpha/lower
	
	
	def run(self,env):
		for i in range(1,self.T+1): # important: Start from 1, because current time is used in the formula.
			
			# Step 1:
			
			s_zero = set()
			omegas2 = self.omegas.copy()# Actually part of step 2.
			
			total_omega = 0
			for omega in self.omegas:
				total_omega += omega
			if max(omega) >= (1/self.k - self.gamma/self.K)*total_omega/(1-self.gamma):
			
				"""Obeservation: The quotient shall be target. It is alpha divided by the sum of all weights,
				but with any individual weight being capped by alpha. That means for a large alpha (highest weight or more), the lower
				part will just be the sum of weights, wheras the upper part and so the full quotient increases as alpha does.
				For a small alpha (lowest weight or less), the lower part will be K times alpha, therefore the quotient will be
				alpha/K*alpha, which is 1/K. Changing alpha has no effect until it is increased beyond the first weight, which will
				then "get stuck" at its value and not increase any more. So the lower part will be less than K*alpha. This effect
				drastifies whenever another weight gets stuck. Therefore, the quotient will simply increase steadily as alpha does (once
				it is higher than the lowest weight) and not show any weird behaviour.
				
				This leads to the following strategy:
					Set alpha to the highest weight.
					As long as the quotient is bigger than the target, set alpha to the next lower weight.
					Solve the equation where all weights passed are alpha and the rest are fixed to their actual value.
				"""
				alpha = max(self.omegas)
				# initially, all weights are included with their actual value because alpha is so high.
				included = self.omegas.copy()
				# And 0 weights are NOT included.
				len_excluded = 0
				
				# A binary search would improve this part, but the overall runtime per timestep is said to be O(K) anyway...
				while calcResult(alpha) > target:
					include.remove(max(included))
					len_excluded += 1
					alpha = max(included)
				# We now have a lower bound for alpha.
				# We also know that any weights that are not within included are capped to alpha, because our upper bound is
				# the lowest weight not withing included, or infinity if there is no such.
				
				# The sum of the weights still included is fixed.
				fixed = 0
				for omega in included:
					fixed += omega
				
				# Now simply solve the equation alpha/(fixed+alpha*len_excluded) = target, which is equal to
				# alpha/target = fixed + alpha*len_excluded, which is equal to
				# alpha/target - alpha*len_excluded = fixed, which is equal to
				# alpha * ( 1/target - len_excluded) = fixed, which is equal to
				alpha = fixed / (1/target - len_excluded)
				
				for i in len(self.omegas):
					if self.omegas[i] >= alpha:
						s_zero.add(i)
						omegas2[i] = alpha
			
			# Step 3:
			propabilities = list()
			total_omega2 = 0
			for omega2 in omegas2:
				total_omega2 += omega2
			for omega2 in omegas2:
				propabilities.append(self.k*((1-self.gamma)*omega2/total_omega2 + self.gamma/self.K))
			
			# Step 4:
			selection = depRound(self.k, propabilities)
			
			# Step 5 and 6:
			x_hat = list()
			for i in num_arm:
				if i in selection:
					x_hat.append(env.feedback(i)[0] / propabilities[i])
				else:
					x_hat.append(0)
			
			for i in num_arm:
				if i not in s_zero:
					self.omegas[i] = self.omegas[i] * np.exp(self.k*gamma*x_hat/self.K)
			
			
			# For analysis only:
			#self.sum_regret += (best - reward)
			#self.avg_regret[i-1] += self.sum_regret/(i)
			#self.cum_regret[i-1] += self.sum_regret
	
	def get_avg_rgt(self):
		return self.avg_regret
	
	def get_cum_rgt(self):
		return self.cum_regret
	
	def clear(self):
		# (Re-)init the omega values.
		self.omegas = [1]*self.K
		
		# And the rest.
		self.propabilities = [0]*self.K
		self.sum_regret = 0
		self.avg_regret = np.zeros(self.T)
		self.cum_regret = np.zeros(self.T)
