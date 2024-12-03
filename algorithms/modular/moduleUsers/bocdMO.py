import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.moduleUsers.basicMultiObjective import BasicMultiObjective
from algorithms.modular.selectionModules.MO_OGDE_Module import MO_OGDE_Module
from algorithms.modular.adaptionModules.bocdModule import BOCDModule
from helpers.gini import gini

class BOCD_MO(BasicMultiObjective):
	def __init__(self,T,num_arm, num_objectives, delta, gini_weights):
		self.selection_module = MO_OGDE_Module(T,num_arm, num_objectives, delta, gini_weights)
		self.adaption_module = BOCDModule(self.selection_module, num_arm, T, num_objectives)
		self.weights = gini_weights
		
		self.T = T
		self.clear()
