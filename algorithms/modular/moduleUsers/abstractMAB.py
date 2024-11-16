class AbstractMAB:
	def __init__(self,T):
		self.T = T
		self.clear()
	
	def run(self,env):
		for t in range(0,self.T):
			arm = self.selection_module.suggestArm()
			reward, optimal_reward = env.feedback(arm)
			self.selection_module.thisHappened(arm, reward, t)
			self.adaption_module.thisHappened(arm, reward, t)
			
			# For performance analysis
			self.sum_rgt += (optimal_reward - reward)
			self.avg_rgt[t] += self.sum_rgt/(t+1)
			self.cum_rgt[t] += self.sum_rgt
	
	def get_avg_rgt(self):
		return self.avg_rgt
	
	def get_cum_rgt(self):
		return self.cum_rgt
	
	def clear(self):
		self.sum_rgt = 0
		self.avg_rgt = [0]*self.T
		self.cum_rgt = [0]*self.T
		
		self.selection_module.fullReset()
		self.adaption_module.fullReset()
