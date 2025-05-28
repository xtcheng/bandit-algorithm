import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.moduleUsers.abstractMAB import AbstractMAB
from algorithms.modular.selectionModules.linUcbModule import LinUCBModule
from algorithms.modular.adaptionModules.nullAdaptionModule import NullAdaptionModule

class LinUCB(AbstractMAB):
	def __init__(self,T,num_arm, num_features, alpha):
		self.selection_module = LinUCBModule(T,num_arm,num_features,alpha)
		self.adaption_module = NullAdaptionModule(self.selection_module)
		super().__init__(T)
	
	def run(self,env):
		for t in range(0,self.T):
			self.selection_module.knowArmFeatures(env.getArmFeatures()) # New!
			print("Learned", self.selection_module.arm_features)
			arm = self.selection_module.suggestArm()
			print("Used", self.selection_module.arm_features)
			reward, optimal_reward = env.feedback(arm)
			self.selection_module.thisHappened(arm, reward, t)
			self.adaption_module.thisHappened(arm, reward, t)
			print("Used", self.selection_module.arm_features)
			
			# For performance analysis
			self.sum_rgt += (optimal_reward - reward)
			self.avg_rgt[t] += self.sum_rgt/(t+1)
			self.cum_rgt[t] += self.sum_rgt
