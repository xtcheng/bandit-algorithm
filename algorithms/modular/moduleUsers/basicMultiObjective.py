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
	
	def gini_cost(self,arm,env):
		gc = 0
		mu = env.getMu()
		costs = sorted(mu[arm], reverse=True)
		for i in range(len(costs)):
			gc += self.weights[i] * costs[i]
		return gc
	
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
			#ins = self.gini_cost(arm,env) - optimal_costs
			
			self.sum_rgt += ins
			self.sum_pto_rgt += env.getParetoRegret(arm)
			self.avg_rgt[t] += self.sum_rgt / (t+1)
			self.cum_rgt[t] += self.sum_rgt
			self.eff_rgt[t] += gini_avg - optimal_costs
			self.cum_pto_rgt[t] += self.sum_pto_rgt 
			self.metrics["Effective Nash Regret"][t] = optimal_nash - nash_avg
		self.epilogue()
	
	def epilogue(self):
		#print(self.selection_module.mu)
		#print("Settled on this mix:")
		#print(self.selection_module.current_mix)
		pass
	
	
	def get_eff_rgt(self):
		return self.eff_rgt
	
	def get_pto_rgt(self):
		return self.cum_pto_rgt
	
	def getMetric(self, key):
		return self.metrics[key]
	
	def listMetrics(self):
		return {"Effective Nash Regret"}
	
	def clear(self):
		# Also set the effective regret because we have it.
		# And Pareto regret.
		self.eff_rgt = [0]*self.T
		self.cum_pto_rgt = [0]*self.T
		self.sum_pto_rgt = 0
		self.historyContainer.fullReset()
		self.metrics = dict()
		self.metrics["Effective Nash Regret"] = [0]*self.T
		
		super().clear()
