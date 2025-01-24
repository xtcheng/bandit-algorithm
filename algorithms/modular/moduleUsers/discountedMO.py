import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.moduleUsers.basicMultiObjective import BasicMultiObjective
from algorithms.modular.selectionModules.MO_OGDE_Module import MO_OGDE_Module
from algorithms.modular.adaptionModules.discountModule import DiscountModule
from algorithms.modular.historyContainerMO import HistoryContainerMO
from helpers.gini import gini

class DiscountedMO(BasicMultiObjective):
	def __init__(self,T,num_arm, num_objectives, delta, gini_weights, gamma):
		self.historyContainer = HistoryContainerMO(num_arm, num_objectives)
		self.selection_module = MO_OGDE_Module(self.historyContainer, T, num_arm, num_objectives, delta, gini_weights)
		self.adaption_module = DiscountModule(self.historyContainer, num_arm, gamma)
		self.weights = gini_weights
		
		self.T = T
		self.clear()
