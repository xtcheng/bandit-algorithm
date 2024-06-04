"""
At each time step, this environment picks a random
number equal to the index of one of the arms.
If the arm the agent chooses has a smaller index, return 0,
otherwise return 1.
"""
import numpy as np

class env_adverse1:
	def __init__(self,num_arm, noise=0):
		self.num_arm = num_arm
		self.noise = noise
		self.rng = np.random.default_rng()
	
	def feedback(self,arm):
		golden_index = self.rng.integers(0, self.num_arm)
		reward = 0
		if arm >= golden_index:
			reward = 1
		
		# Does it make sense to add noise in a nonstochastic env?
		# Not sure, so consider case where there is no noise.
		if self.noise != 0:
			reward += self.noise.sample_trunc()
		
		# Due to definition, there is always an arm with
		# reward=1 which is optimal (neglecting noise).
		return reward, 1

"""
ad = env_adverse1(5)
for i in [0,2,4]:
	print("Arm", i)
	for j in range(0,30):
		print(ad.feedback(i)[0], end=" ")
	print("\n")
"""
