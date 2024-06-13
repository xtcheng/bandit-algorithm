import numpy as np

class exp3M:
	def __init__(self,T,num_arm, k, gamma=0.2):
		self.T = T
		self.K = num_arm
		self.k = k
		self.gamma = gamma
		self.rng = np.random.default_rng()
		self.clear()
	
	
	def calcResult(self, alpha):
		# Calculate the left part of the equation to be satisfied using a given alpha.
		lower = 0
		for omega in self.omegas:
			if omega >= alpha:
				lower += alpha
			else:
				lower += omega
		return alpha/lower
	
	
	def depRound(self, k, values):
		# Perform the dependency rounding and return the indexes of the 1s as a set.
		
		# The set of values that are neither 1 nor 0.
		undecided = list()
		for i in range(len(values)):
			if values[i] != 1 and values[i] != 0:
				undecided.append(i)
		
		while len(undecided) > 0:
			if len(undecided) == 1:
				raise AssertionError("Cannot select 2 distinct elements from a set of size 1.")
			
			# Select 2 distinct indexes
			i = 0
			j = 0
			while i==j:
				i, j = np.random.choice(undecided, 2)
			
			# Set alpha and beta as defined
			alpha = min((1-values[i], values[j]))
			beta = min((values[i], 1-values[j]))
			
			# Update the values as defined, but also update undecided
			if self.rng.random() < beta/(alpha+beta):
				values[i] = values[i] + alpha
				values[j] = values[j] - alpha
			else:
				values[i] = values[i] - beta
				values[j] = values[j] + beta
			if values[i] == 1 or values[i] == 0:
				undecided.remove(i)
			if values[j] == 1 or values[j] == 0:
				undecided.remove(j)
			
			#print(values)
			#print(undecided)
			
			"""Rounding errors can and will occur. k is an integer, so the total propability mass can
			be reshifted so that only 1s and 0s are left at the end. Which means that when
			there is exactly 1 element left that is not 1 or 0, it is the result of accumulated
			Rounding errors. Catch this case."""
			if len(undecided) == 1:
				values[undecided[0]] = np.round(values[undecided[0]])
				undecided.pop()
		
		# Now exactly k elements are 1. Return these.
		ret = set()
		for i in range(len(values)):
			if values[i] == 1:
				ret.add(i)
		if len(ret) != k:
			raise AssertionError("The number of elements picked is "+str(len(ret))+" instead of "+str(k)+".")
		return ret
	
	
	def run(self,env):
		for timestep in range(1,self.T+1): # important: Start from 1, because current time is used in the formula.
			
			# Step 1:
			
			s_zero = set()
			omegas2 = self.omegas.copy()# Actually part of step 2.
			
			total_omega = 0
			for omega in self.omegas:
				total_omega += omega
			if max(self.omegas) >= (1/self.k - self.gamma/self.K)*total_omega/(1-self.gamma):
			
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
				target = (1/self.k - self.gamma/self.K) / (1 - self.gamma)
				alpha = max(self.omegas)
				# initially, all weights are included with their actual value because alpha is so high.
				included = self.omegas.copy()
				# And 0 weights are NOT included.
				len_excluded = 0
				
				# A binary search would improve this part, but the overall runtime per timestep is said to be O(K) anyway...
				while self.calcResult(alpha) > target:
					included.remove(max(included))
					len_excluded += 1
					alpha = max(included)
				"""We now have a lower bound for alpha.
				We also know that any weights that are not within included are capped to alpha,
				because our upper bound is the lowest weight not withing included,
				or infinity if there is no such."""
				
				# The sum of the weights still included is fixed.
				fixed = 0
				for omega in included:
					fixed += omega
				
				"""Now simply solve the equation alpha/(fixed+alpha*len_excluded) = target, which is equal to
				alpha/target = fixed + alpha*len_excluded, which is equal to
				alpha/target - alpha*len_excluded = fixed, which is equal to
				alpha * ( 1/target - len_excluded) = fixed, which is equal to"""
				alpha = fixed / (1/target - len_excluded)
				
				for i in range(len(self.omegas)):
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
			selection = self.depRound(self.k, propabilities)
			
			# Step 5:
			rewards, best = env.feedbackMult(selection)
			#print("Received rewards", rewards, "best total would be", best)
			total_reward = 0
			for reward in rewards:
				total_reward += reward
			
			# Step 6:
			x_hat = list()
			for i in range(self.K):
				if i in selection:
					x_hat.append(rewards[i] / propabilities[i])
				else:
					x_hat.append(0)
			
			for i in range(self.K):
				if i not in s_zero:
					self.omegas[i] = self.omegas[i] * np.exp(self.k*self.gamma*x_hat[i]/self.K)
			
			# For analysis only:
			self.sum_regret += (best - total_reward)
			self.avg_regret[timestep-1] += self.sum_regret/timestep
			self.cum_regret[timestep-1] += self.sum_regret
	
	
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
