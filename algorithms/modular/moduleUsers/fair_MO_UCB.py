import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.moduleUsers.basicMultiObjective import BasicMultiObjective
from algorithms.modular.selectionModules.fair_UCB_Module import Fair_UCB_Module
from algorithms.modular.adaptionModules.bocdModule import BOCDModule
from algorithms.modular.adaptionModules.nullAdaptionModule import NullAdaptionModule
from algorithms.modular.historyContainerMO import HistoryContainerMO

class Fair_MO_UCB(BasicMultiObjective):
	def __init__(self,T,num_arm, num_objectives, delta, gini_weights):
		self.historyContainer = HistoryContainerMO(num_arm, num_objectives)
		self.selection_module = Fair_UCB_Module(self.historyContainer, T, num_arm, num_objectives, delta)
		self.adaption_module = NullAdaptionModule(self.historyContainer)
		self.weights = gini_weights
		
		self.T = T
		self.clear()
