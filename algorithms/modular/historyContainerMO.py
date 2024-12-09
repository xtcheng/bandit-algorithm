import numpy as np

class HistoryContainerMO:
	def __init__(self, num_arm, num_objectives):
		self.num_arm = num_arm
		self.num_objectives = num_objectives
		self.users = []
		self.fullReset()
	
	
	def thisHappened(self, arm, reward, t):
		# To update the information after a pull. It's still called reward, but it are actually costs this time.
		self.sum_mu[arm] += reward
		self.num_play[arm] += 1
		self.current_turn += 1
		
		for i in range(self.num_objectives):
			self.mu[arm][i] = self.sum_mu[arm][i] / self.num_play[arm]
		for a in range(self.num_arm):
			# Refresh all because they might have been changed by the adaption module.
			self.sorted_mu[a] = sorted(self.mu[a], reverse=True)
		
		# Let all experts do their own calculation
		for user in self.users:
			user.updateMix()
	
	def fullReset(self):
		# Reset everything to the start.
		self.sum_mu = np.zeros((self.num_arm, self.num_objectives))
		self.mu = np.zeros((self.num_arm, self.num_objectives))
		self.sorted_mu = np.zeros((self.num_arm, self.num_objectives))
		self.num_play = [0]*self.num_arm
		self.current_turn = 1
		for user in self.users:
			user.fullReset()
	
	def resetArm(self, arm):
		# Reset only one arm.
		self.sum_mu[arm] = np.zeros(self.num_objectives)
		self.current_turn -= self.num_play[arm]
		self.num_play[arm] = 0
	
	def register(self, user):
		self.users.append(user)
