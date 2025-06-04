import numpy as np
import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.selectionModules.abstractSelectionModule import AbstractSelectionModule

class ThompsonModule(AbstractSelectionModule):
	def __init__(self,T,num_arm):
		super().__init__(T, num_arm)
	
	def thisHappened(self, arm, reward, timestep):
		# Turn a stochastic reward into a bernoulli reward. Interestingly, as random.random is in [0, 1), for reward=1 this will always yield 1 and for reward=0 always 0, so this step is transparent if we are already dealing with a bernoulli bandit. This means we do need to tell apart a stochastic from a bernoulli bandit and can simply always perform this step.
		# Note that we store the number of successes and of total plays on that arm, instead of successes and failures. This holds equal information, but ensures compatibility with our adaption modules. (However, note that the discount module will create non-int entries, which is not supposed to happen in the original thompson sampling strategy.)
		if np.random.random() < reward:
			self.sum_mu[arm] += 1
		self.num_play[arm] += 1
	
	def suggestArm(self):
		# Simulate the probabilities of drawing success, by plugging the information we have into the beta function.
		samples = [0]*self.num_arm
		for arm in range(self.num_arm):
			samples[arm] = np.random.beta(self.sum_mu[arm]+1, self.num_play[arm]-self.sum_mu[arm]+1)
		return np.argmax(samples)
