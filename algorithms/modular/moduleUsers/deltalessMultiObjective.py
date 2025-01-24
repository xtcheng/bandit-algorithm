from algorithms.modular.moduleUsers.basicMultiObjective import BasicMultiObjective
from algorithms.modular.selectionModules.deltaless_MO_OGDE_Module import Deltaless_MO_OGDE_Module
from algorithms.modular.adaptionModules.nullAdaptionModule import NullAdaptionModule
from algorithms.modular.historyContainerMO import HistoryContainerMO

class DeltalessMultiObjective(BasicMultiObjective):
	def __init__(self,T,num_arm, num_objectives, gini_weights):
		self.historyContainer = HistoryContainerMO(num_arm, num_objectives)
		self.selection_module = Deltaless_MO_OGDE_Module(self.historyContainer, T, num_arm, num_objectives, gini_weights)
		self.adaption_module = NullAdaptionModule(self.historyContainer)
		self.weights = gini_weights
		
		self.T = T
		self.clear()
