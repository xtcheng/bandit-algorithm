import numpy as np
from algorithms.modular.moduleUsers.expertsMultiObjective import ExpertsMultiObjective
from algorithms.modular.selectionModules.PL_MO_OGDE_Module import PL_MO_OGDE_Module
from algorithms.modular.adaptionModules.discountModule import DiscountModule
from algorithms.modular.historyContainerMO import HistoryContainerMO
import math

class DiscountedExpertsMultiObjective(ExpertsMultiObjective):
	def __init__(self,T,num_arm, num_objectives, gini_weights, gamma):
		self.historyContainer = HistoryContainerMO(num_arm, num_objectives)
		self.experts = []
		self.num_experts = round(1/2 * math.ceil(math.log2(2*T + 1) ) + 1)
		print("Using", self.num_experts, "experts.")
		for q in range(self.num_experts):
			learning_factor = 2**q
			self.experts.append(PL_MO_OGDE_Module(self.historyContainer, T, num_arm, num_objectives, learning_factor, gini_weights))
		self.adaption_module = DiscountModule(self.historyContainer, num_arm, gamma)
		self.weights = gini_weights
		
		self.T = T
		self.clear()
