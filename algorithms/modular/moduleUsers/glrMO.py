import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.moduleUsers.basicMultiObjective import BasicMultiObjective
from algorithms.modular.selectionModules.MO_OGDE_Module import MO_OGDE_Module
from algorithms.modular.adaptionModules.glrModule import GLRModule
from helpers.gini import gini

class GLR_MO(BasicMultiObjective):
	def __init__(self,T,num_arm, num_objectives, delta, delta2, global_restart, lazyness, gini_weights):
		self.selection_module = MO_OGDE_Module(T,num_arm, num_objectives, delta, gini_weights)
		self.adaption_module = GLRModule(self.selection_module, num_arm, delta2, global_restart, lazyness, num_objectives)
		self.weights = gini_weights
		
		self.T = T
		self.clear()
