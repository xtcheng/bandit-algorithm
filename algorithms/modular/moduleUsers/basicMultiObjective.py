import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.moduleUsers.abstractMAB import AbstractMAB
from algorithms.modular.selectionModules.MO_OGDE_Module import MO_OGDE_Module
from algorithms.modular.adaptionModules.nullAdaptionModule import NullAdaptionModule
from algorithms.modular.historyContainerMO import HistoryContainerMO
from helpers.gini import gini

class BasicMultiObjective(AbstractMAB):
	def __init__(self,T,num_arm, num_objectives, delta, gini_weights):
		self.historyContainer = HistoryContainerMO(num_arm, num_objectives)
		self.selection_module = MO_OGDE_Module(self.historyContainer, T, num_arm, num_objectives, delta, gini_weights)
		self.adaption_module = NullAdaptionModule(self.historyContainer)
		self.weights = gini_weights
		
		super().__init__(T)
	
	
	# run needs to be redifined because of the reward being multi-dimensional and there is no optimal action, but an optimal mix.
	def run(self,env):
		total = 0 # Total costs
		for t in range(0,self.T):
			arm = self.selection_module.suggestArm()
			reward, optimal_costs = env.feedback(arm)
			self.historyContainer.thisHappened(arm, reward, t)
			self.adaption_module.thisHappened(arm, reward, t)
			
			
			# For performance analysis
			
			# Regret
			total += reward
			ar = [self.weights]
			ar.append(total / (t+1))
			gini_avg = gini([1], ar)
			
			
			# Instantaneous regret
			ar = [self.weights]
			for m in env.getMu():
				# Of course you must not use such direct access outside the analysis.
				ar.append(m)
			ins = gini(self.selection_module.current_mix, ar) - optimal_costs
			
			
			self.sum_rgt += ins
			self.avg_rgt[t] = self.sum_rgt / (t+1)
			self.cum_rgt[t] = self.sum_rgt
			self.eff_rgt[t] = gini_avg - optimal_costs
		#print(self.selection_module.mu)
		#print("Settled on this mix:")
		#print(self.selection_module.current_mix)
	
	
	def get_eff_rgt(self):
		return self.eff_rgt
	
	
	def clear(self):
		# Also set the effective regret because we have it.
		self.eff_rgt = [0]*self.T
		self.historyContainer.fullReset()
		super().clear()
