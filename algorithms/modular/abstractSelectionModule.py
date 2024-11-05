class AbstractSelectionModule:
	def __init__(self,T,num_arm):
		self.T = T
		self.num_arm = num_arm
		self.fullReset()
	
	def suggestArm(self):
		# Perform the specific selection strategy with the current knowledge and return what arm is to be pulled.
		pass
	
	
	def thisHappened(self, arm, reward, timestep):
		# To update the information after a pull.
		self.sum_mu[arm] += reward
		self.num_play[arm] += 1
		self.current_turn += 1
	
	def fullReset(self):
		# Reset everything to the start.
		self.sum_mu = [0]*self.num_arm
		self.num_play = [0]*self.num_arm
		self.relative_start = [1]*self.num_arm
		self.current_turn = 1
	
	def resetArm(self, arm):
		# Reset only one arm.
		self.sum_mu[arm] = 0
		self.num_play[arm] = 0
		self.relative_start[arm] = 1
