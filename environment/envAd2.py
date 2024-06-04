"""
At each time step, this environment picks a random
number equal to the index of one of the arms.
If the agent chooses this very index, return 1,
otherwise return 0.
"""
import numpy as np

class env_adverse2:
	def __init__(self,num_arm, noise=0):
		self.num_arm = num_arm
		self.noise = noise
		self.rng = np.random.default_rng()
	
	def feedback(self,arm):
		golden_index = self.rng.integers(0, self.num_arm)
		reward = 0
		if arm == golden_index: #The only difference to env_adverse1.
			reward = 1
		
		# Does it make sense to add noise in a nonstochastic env?
		# Not sure, so consider case where there is no noise.
		if self.noise != 0:
			reward += self.noise.sample_trunc()
		
		# Due to definition, there is always an arm with
		# reward=1 which is optimal (neglecting noise).
		return reward, 1


"""
for i in [2,5,50]:
	ad = env_adverse2(i)
	print(i, "Arms")
	for j in range(0,30):
		print(ad.feedback(0)[0], end=" ")
	print("\n")
"""
