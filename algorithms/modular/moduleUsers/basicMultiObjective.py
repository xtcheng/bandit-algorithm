import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.moduleUsers.abstractMAB import AbstractMAB
from algorithms.modular.selectionModules.MO_OGDE_Module import MO_OGDE_Module
from algorithms.modular.adaptionModules.nullAdaptionModule import NullAdaptionModule
from algorithms.modular.historyContainerMO import HistoryContainerMO
from helpers.gini import gini
from helpers.nash import nash
from helpers.commonTools import costs2rewards

class BasicMultiObjective(AbstractMAB):
	def __init__(self,T,num_arm, num_objectives, delta, gini_weights):
		self.historyContainer = HistoryContainerMO(num_arm, num_objectives)
		self.selection_module = MO_OGDE_Module(self.historyContainer, T, num_arm, num_objectives, delta, gini_weights)
		self.adaption_module = NullAdaptionModule(self.historyContainer)
		self.weights = gini_weights
		
		super().__init__(T)
	
	
	def pullArm(self, t, env):
		# Needs to be in an extra function because this part is different in the expert version.
		arm = self.selection_module.suggestArm()
		reward, optimal_costs = env.feedback(arm)
		self.historyContainer.thisHappened(arm, reward, t)
		self.adaption_module.thisHappened(arm, reward, t)
		self.current_mix = self.selection_module.current_mix
		return reward, optimal_costs, arm
	
	
	# run needs to be redifined because of the reward being multi-dimensional and there is no optimal action, but an optimal mix.
	def run(self,env):
		total = 0 # Total costs
		for t in range(0,self.T):
			optimal_nash = env.getOptimalNash()
			reward, optimal_costs, arm = self.pullArm(t, env)
			
			# For performance analysis
			
			# Regret
			total += reward
			ar = [self.weights]
			ar.append(total / (t+1))
			gini_avg = gini([1], ar)
			
			ar = [costs2rewards(total/(t+1))]
			nash_avg = nash([1], ar)
			
			# Instantaneous regret
			ar = [self.weights]
			for m in env.getMu():
				# Of course you must not use such direct access outside the analysis.
				ar.append(m)
			ins = gini(self.current_mix, ar) - optimal_costs
			
			
			self.sum_rgt += ins
			self.sum_pto_rgt += env.getParetoRegret(arm)
			self.avg_rgt[t] = self.sum_rgt / (t+1)
			self.cum_rgt[t] = self.sum_rgt
			self.eff_rgt[t] = gini_avg - optimal_costs
			self.avg_pto_rgt[t] = self.sum_pto_rgt / (t+1)
			self.metrics["Effective Nash Regret"][t] = optimal_nash - nash_avg
		self.epilogue()
	
	def epilogue(self):
		#print(self.selection_module.mu)
		#print("Settled on this mix:")
		#print(self.selection_module.current_mix)
		pass
	
	
	def getMetric(self, key):
		if key == "Effective Regret":
			return self.eff_rgt
		if key == "Cumulative Pareto Regret":
			return self.avg_pto_rgt
		return self.metrics[key]
	
	def listMetrics(self):
		return {"Effective Nash Regret", "Effective Regret", "Cumulative Pareto Regret"}
	
	def clear(self):
		# Also set the effective regret because we have it.
		# And Pareto regret.
		self.eff_rgt = [0]*self.T
		self.avg_pto_rgt = [0]*self.T
		self.sum_pto_rgt = 0
		self.historyContainer.fullReset()
		self.metrics = dict()
		self.metrics["Effective Nash Regret"] = [0]*self.T
		
		super().clear()
