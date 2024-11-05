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
		total1 = 0 # Total costs since start
		total2 = 0 # Total costs since start or breakpoint
		total_mix1 = 0 # Sum of all alphas the selection module uses over time.
		total_mix2 = 0 # Sum of all alphas the selection module uses over time, since start or breakpoint.
		since_break = 0 # steps since start or breakpoint
		
		for t in range(0,self.T):
			arm = self.selection_module.suggestArm()
			reward, optimal_costs = env.feedback(arm)
			self.selection_module.thisHappened(arm, reward, t)
			self.adaption_module.thisHappened(arm, reward, t)
			
			
			# For performance analysis
			
			# Breakpoint?
			if env.turns in env.breakpoints:
				total2 = 0
				total_mix2 = 0
				since_break = 0
			since_break += 1
			
			# Regret
			total1 += reward
			total2 += reward
			
			ar = [self.weights]
			ar.append(total1 / (t+1))
			gini_avg1 = gini([1], ar)
			
			ar[-1] = total2 / since_break
			gini_avg2 = gini([1], ar)
			
			
			# Pseudo regret
			ar = [self.weights]
			for m in env.getMu():
				# Of course you must not use such direct access outside the analysis.
				ar.append(m)
			# Keep track of how the alpha developes
			total_mix1 += self.selection_module.current_mix
			total_mix2 += self.selection_module.current_mix
			pseudo1 = gini(total_mix1 / (t+1), ar)
			pseudo2 = gini(total_mix2 / since_break, ar)
			
			# Instantaneous regret
			ins = gini(self.selection_module.current_mix, ar) - optimal_costs
			
			
			self.sum_rgt += ins
			self.avg_rgt[t] = self.sum_rgt / (t+1)
			self.cum_rgt[t] = self.sum_rgt
			self.psd_rgt1[t] = pseudo1 - optimal_costs
			self.psd_rgt2[t] = pseudo2 - optimal_costs
			self.rgt1[t] = gini_avg1 - optimal_costs
			self.rgt2[t] = gini_avg2 - optimal_costs
		#print(self.selection_module.mu)
		#print("Settled on this mix:")
		#print(self.selection_module.current_mix)
	
	
	def get_psd_rgt1(self):
		return self.psd_rgt1
	
	def get_psd_rgt2(self):
		return self.psd_rgt2
	
	def get_rgt1(self):
		return self.rgt1
	
	def get_rgt2(self):
		return self.rgt2
	
	
	def clear(self):
		# Also set the Pseudo regret because we have it.
		self.psd_rgt1 = [0]*self.T
		self.psd_rgt2 = [0]*self.T
		self.rgt1 = [0]*self.T
		self.rgt2 = [0]*self.T
		super().clear()
