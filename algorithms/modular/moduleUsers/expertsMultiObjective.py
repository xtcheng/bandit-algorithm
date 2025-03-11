import numpy as np
from algorithms.modular.moduleUsers.basicMultiObjective import BasicMultiObjective
from algorithms.modular.selectionModules.PL_MO_OGDE_Module import PL_MO_OGDE_Module
from algorithms.modular.adaptionModules.nullAdaptionModule import NullAdaptionModule
from algorithms.modular.historyContainerMO import HistoryContainerMO
from scipy import optimize
from helpers.gini import gini
import math

class ExpertsMultiObjective(BasicMultiObjective):
	def __init__(self,T,num_arm, num_objectives, gini_weights):
		self.historyContainer = HistoryContainerMO(num_arm, num_objectives)
		self.experts = []
		self.num_experts = round(1/2 * math.ceil(math.log2(2*T + 1) ) + 1) # or should I include the 1/2 in the floor to avoid the rounding?
		print("Using", self.num_experts, "experts.")
		for q in range(self.num_experts):
			learning_factor = 2**q
			self.experts.append(PL_MO_OGDE_Module(self.historyContainer, T, num_arm, num_objectives, learning_factor, gini_weights))
		self.adaption_module = NullAdaptionModule(self.historyContainer)
		self.weights = gini_weights
		
		self.T = T
		self.clear()
	
	
	def pullArm(self, t, env):
		self.current_mix = np.array([0.0]*self.historyContainer.num_arm)
		opinions = []
		for i in range(self.num_experts):
			opinions.append(self.experts[i].current_mix.tolist()) # Use tolist to use buildin type, numpy will totally screw up if we keep a reference to that array we later change somewhere!!
			self.current_mix += self.experts[i].current_mix * self.expert_weights[i]
		self.current_mix /= sum(self.current_mix)
		
		if 0 in self.historyContainer.num_play:
			arm = self.historyContainer.num_play.index(0)
			opinions = 0
		
		else:
			# Select a random arm according to the distribution
			arm = np.random.choice(range(self.historyContainer.num_arm), p=(self.current_mix))
		reward, optimal_costs = env.feedback(arm)
		self.historyContainer.thisHappened(arm, reward, t)
		if opinions != 0:
			# If we used the experts' opinions, rate them.
			
			self.calcOptimalCosts()
			losses = []
			epsilon = 1
			for i in range(self.num_experts):
				ar = [self.weights]
				for m in self.historyContainer.mu:
					ar.append(m)
				# The loss is what this alpha would yield if our estimates were 100% correct, minus what the best costs for these estimates would be.
				raw_loss = gini(opinions[i], ar) - self.opt
				losses.append(self.expert_weights[i] * np.exp(-epsilon * raw_loss))
				#losses.append(self.expert_weights[i] * np.exp(-raw_loss))
			loss_sum = sum(losses)
			
			# Update the expert weights.
			for i in range(self.num_experts):
				self.expert_weights[i] = losses[i] / loss_sum
		self.adaption_module.thisHappened(arm, reward, t)
		return reward, optimal_costs, arm
	
	
	def epilogue(self):
		print("Final expert weights:", self.expert_weights)
	
	def calcOptimalCosts(self):
		# Calculate the optimal costs, but based only what we internally know.
		ar = [self.weights]
		for m in self.historyContainer.mu:
			ar.append(m)
		con = [{'type':'eq', 'fun': (lambda x : sum(x)-1)}]
		result = optimize.minimize(gini, args=ar, x0=[1/self.historyContainer.num_arm]*self.historyContainer.num_arm, constraints=con, bounds=[(0,1)]*self.historyContainer.num_arm)
		self.opt = result.fun
	
	def clear(self):
		self.eff_rgt = [0]*self.T
		self.sum_rgt = 0
		self.avg_rgt = [0]*self.T
		self.cum_rgt = [0]*self.T
		self.avg_pto_rgt = [0]*self.T
		self.sum_pto_rgt = 0
		
		self.metrics = dict()
		self.metrics["Effective Nash Regret"] = [0]*self.T
		
		self.historyContainer.fullReset()
		self.adaption_module.fullReset()
		
		self.expert_weights = []
		for i in range(1, len(self.experts)+1):
			self.expert_weights.append( ((len(self.experts)+1) / len(self.experts)) / (i*(i+1)) )
		self.current_mix = np.array([1/self.historyContainer.num_arm]*self.historyContainer.num_arm)