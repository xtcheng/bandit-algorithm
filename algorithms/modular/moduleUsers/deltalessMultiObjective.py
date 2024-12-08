from algorithms.modular.moduleUsers.basicMultiObjective import BasicMultiObjective
from algorithms.modular.selectionModules.deltaless_MO_OGDE_Module import Deltaless_MO_OGDE_Module
from algorithms.modular.adaptionModules.nullAdaptionModule import NullAdaptionModule

class DeltalessMultiObjective(BasicMultiObjective):
	def __init__(self,T,num_arm, num_objectives, gini_weights):
		self.selection_module = Deltaless_MO_OGDE_Module(T,num_arm, num_objectives, gini_weights)
		self.adaption_module = NullAdaptionModule(self.selection_module)
		self.weights = gini_weights
		
		self.T = T
		self.clear()
