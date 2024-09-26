import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.abstractMAB import AbstractMAB
from algorithms.modular.MO_OGDE_Module import MO_OGDE_Module
from algorithms.modular.nullAdaptionModule import NullAdaptionModule

class BasicMultiObjective(AbstractMAB):
	def __init__(self,T,num_arm, num_objectives, delta, gini_weights):
		self.selection_module = MO_OGDE_Module(T,num_arm, num_objectives, delta, gini_weights)
		self.adaption_module = NullAdaptionModule(self.selection_module)
		super().__init__(T)
	
	
	# run needs to be redifined because of the reward being multi-dimensional and there is no optimal action, but an optimal mix.
	def run(self,env):
		for t in range(0,self.T):
			arm = self.selection_module.suggestArm()
			reward, optimal_reward = env.feedback(arm)
			self.selection_module.thisHappened(arm, reward)
			self.adaption_module.thisHappened(arm, reward)
			
			"""
			# For performance analysis
			# TODO
			
			self.sum_rgt += (optimal_reward - reward)
			self.avg_rgt[t] += self.sum_rgt/(t+1)
			self.cum_rgt[t] += self.sum_rgt"""
		print(self.selection_module.mu)
		print("Settled on this mix:")
		print(self.selection_module.current_mix)