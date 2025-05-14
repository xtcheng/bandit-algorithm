import numpy as np
import sys
if not "../" in sys.path:
	sys.path.append('../')

class LinUCBModule2:
	def __init__(self,T,num_arm, num_features, alpha):
		self.T = T
		self.num_arm = num_arm
		self.num_features = num_features
		self.alpha = alpha
		self.fullReset()
	
	def suggestArm(self):
		# If some feature has not been explored at all, choose the arm where it is strongest.
		if 0 in self.feature_exploration:
			feature = self.feature_exploration.index(0)
			arm = 0
			for testarm in range(1, self.num_arm):
				if self.arm_features[testarm][feature] > self.arm_features[arm][feature]:
					arm = testarm
			return arm
		
		# Otherwise, calculate UCB as you would for arms and choose the arm that can amplify this best via the dot product.
		ucb = [0]*self.num_features
		for feature in range(self.num_features):
			mu = self.pseudo_sum_mu[feature]/self.feature_exploration[feature]
			ucb[feature] = mu + self.alpha*np.sqrt(np.log(self.pseudo_current_turn)/self.feature_exploration[feature])
		
		best_reward = -1
		best_arm = 0
		for arm in range(self.num_arm):
			reward = np.dot(ucb, self.arm_features[arm])
			if reward > best_reward:
				best_reward = reward
				best_arm = arm
		return best_arm
	
	def thisHappened(self, arm, reward, timestep):
		# Take the full reward and split it onto the features according to their presence in the chosen arm. This is a big simplification, but may work.
		total_features = sum(arm_features[arm])
		self.pseudo_current_turn += total_features
		for feature in range(self.num_features):
			self.feature_exploration[feature] += self.arm_features[arm][feature]
			self.pseudo_sum_mu[feature] += reward  * (self.arm_features[arm][feature] / total_features)
	
	def knowArmFeatures(self, features):
		self.arm_features = features
	
	def fullReset(self):
		self.feature_exploration = [0]*self.num_features
		self.pseudo_sum_mu = [0]*self.num_features
		self.pseudo_current_turn = 0
