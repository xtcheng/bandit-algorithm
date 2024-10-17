import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.abstractMAB import AbstractMAB
from algorithms.modular.MO_OGDE_Module import MO_OGDE_Module
from algorithms.modular.nullAdaptionModule import NullAdaptionModule
from algorithms.modular.gini import gini

class BasicMultiObjective(AbstractMAB):
	def __init__(self,T,num_arm, num_objectives, delta, gini_weights):
		self.selection_module = MO_OGDE_Module(T,num_arm, num_objectives, delta, gini_weights)
		self.adaption_module = NullAdaptionModule(self.selection_module)
		self.weights = gini_weights
		
		super().__init__(T)
	
	
	# run needs to be redifined because of the reward being multi-dimensional and there is no optimal action, but an optimal mix.
	def run(self,env):
		total = 0 # Total costs
		total_mix = 0 # Sum of all alphas the selection module uses over time.
		for t in range(0,self.T):
			arm = self.selection_module.suggestArm()
			reward, optimal_costs = env.feedback(arm)
			self.selection_module.thisHappened(arm, reward)
			self.adaption_module.thisHappened(arm, reward)
			
			
			# For performance analysis
			
			# Regret
			total += reward
			ar = [self.weights]
			ar.append(total / (t+1))
			gini_avg = gini([1], ar)
			
			# Pseudo regret
			ar = [self.weights]
			for m in env.mu:
				# Of course you must not use such direct access outside the analysis.
				ar.append(m)
			
			# Keep track of how the alpha developes
			total_mix += self.selection_module.current_mix
			
			pseudo = gini(total_mix / (t+1), ar)
			
			
			self.sum_rgt += gini_avg - optimal_costs
			self.avg_rgt[t] += gini_avg - optimal_costs
			self.cum_rgt[t] += self.sum_rgt
			self.psd_rgt[t] += pseudo - optimal_costs
		print(self.selection_module.mu)
		print("Settled on this mix:")
		print(self.selection_module.current_mix)
	
	
	def get_psd_rgt(self):
		return self.psd_rgt
	
	
	def clear(self):
		# Also set the Pseudo regret because we have it.
		self.psd_rgt = [0]*self.T
		super().clear()
