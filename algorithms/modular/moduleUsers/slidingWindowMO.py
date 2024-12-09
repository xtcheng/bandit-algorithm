import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.moduleUsers.basicMultiObjective import BasicMultiObjective
from algorithms.modular.selectionModules.MO_OGDE_Module import MO_OGDE_Module
from algorithms.modular.adaptionModules.slidingWindowModule import SlidingWindowModule
from algorithms.modular.historyContainerMO import HistoryContainerMO
from helpers.gini import gini

class SlidingWindowMO(BasicMultiObjective):
	def __init__(self,T,num_arm, num_objectives, delta, gini_weights, window_len):
		self.historyContainer = HistoryContainerMO(num_arm, num_objectives)
		self.selection_module = MO_OGDE_Module(self.historyContainer, T, num_arm, num_objectives, delta, gini_weights)
		self.adaption_module = SlidingWindowModule(self.historyContainer, num_arm, window_len)
		self.weights = gini_weights
		
		self.T = T
		self.clear()
