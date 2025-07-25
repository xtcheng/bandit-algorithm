from environment.envContextual import EnvContextual
import numpy as np

# This actually works the same as the normal environment, with the exception that the expectation values of the rewards, instead of being arbitrary, are calculated from a set of vectors that can be inquired by the agent plus a single vector that is unknown to the agent. The latter (associated with the user) is always fixed, the others (associated with the arms) may be renewed after each timestep.

class EnvContextualFlip(EnvContextual):
	def __init__(self, num_arm, theta, noise, renewing_arms, corruption_level):
		self.corruption_level = corruption_level
		self.clear()
		self.mu_is_corrupted = False
		# No need to consider linkFunction here; we don't pass it and so it defaults to none and will be set to identity.
		super().__init__(num_arm, theta, noise, renewing_arms)
	
	# Anchor the corruption application here, because this is where we set the possible rewards for next turn.
	def renewArms(self, force=False):
		self.current_turn += 1
		if self.renewing_arms or force:
			for i in range(self.num_arm):
				self.arm_features[i] = np.random.uniform(0,1,self.num_features)
		
		# We need to recalculate the (uncorrupted) mu if the arms have changed or in some cases when we want to apply corruption, because the mu might already be corrupted from the previous turn and then we would apply corruption twice. We also need to recalculate mu if there has been corruption in the previous turn.
		if self.renewing_arms or force or self.mu_is_corrupted:
			for i in range(self.num_arm):
				self.mu[i] = np.dot(self.arm_features[i], self.theta)
				self.mu_is_corrupted = False
		if self.corruptionNow():
			self.corruptMu()
			self.mu_is_corrupted = True
	
	def corruptionNow(self):
		# This specific environment simply applies corruption in the first corruption_level steps.
		return self.current_turn <= self.corruption_level
	
	def corruptMu(self):
		eta = 1 # Maybe use something bigger that depends on the number of features / the greatest possible reward?
		for i in range(self.num_arm):
			# Flip the rewards, making good ones bad (and maybe 0) and bad ones good.
			self.mu[i] = max(0, eta - self.mu[i])
	
	def clear(self):
		self.current_turn = 0
