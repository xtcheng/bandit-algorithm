"""
I've been creative: At any point in time, one arm is the best arm
with reward 1, while the reward of any other arm is 1/(d+1)^2, where
d is the distance to the best arm, assuming circular placement.
Example for d:
2 1 0 1 2 3 4 5 4 3

The distribution stays stable for a constant number of draws, after which
a random arm is assigned as the new best, with the others being refreshed accordingly.
"""
import numpy as np

class env_adverse3:
	def __init__(self,num_arm, noise=0, difficulty=20):
		self.num_arm = num_arm
		self.noise = noise
		self.rng = np.random.default_rng()
		self.difficulty = difficulty #For how many draws it stays stable.
		self.progress = 0 #Triggers switch when reaching difficulty.
		self.rewards = [0]*self.num_arm
	
	def feedback(self,arm):
		self.progress += 1
		if self.progress >= self.difficulty:
			golden_index = self.rng.integers(1, self.num_arm)
			self.rewards = [0]*self.num_arm
			self.rewards[golden_index] = 1
			for i in range(0, int(np.ceil(self.num_arm/2))+1):
				for j in (-i, i):
					if self.rewards[(golden_index+j)%self.num_arm] == 0:
						self.rewards[(golden_index+j)%self.num_arm] = 1 / (i+1)**2
			self.progress = 0
			#print(self.rewards)
			
		reward = self.rewards[arm]
		
		# Does it make sense to add noise in a nonstochastic env?
		# Not sure, so consider case where there is no noise.
		if self.noise != 0:
			reward += self.noise.sample_trunc()
		
		# Due to definition, there is always an arm with
		# reward=1 which is optimal (neglecting noise).
		return reward, 1


"""
for i in [2,5,50]:
	ad = env_adverse3(i)
	print(i, "Arms")
	for j in range(0,30):
		print(ad.feedback(0)[0], end=" ")
	print("\n")
"""
