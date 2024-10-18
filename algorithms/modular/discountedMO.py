import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.basicMultiObjective import BasicMultiObjective
from algorithms.modular.MO_OGDE_Module import MO_OGDE_Module
from algorithms.modular.discountModule import DiscountModule
from algorithms.modular.gini import gini

class DiscountedMO(BasicMultiObjective):
	def __init__(self,T,num_arm, num_objectives, delta, gini_weights, gamma):
		self.selection_module = MO_OGDE_Module(T,num_arm, num_objectives, delta, gini_weights)
		self.adaption_module = DiscountModule(self.selection_module, num_arm, gamma)
		self.weights = gini_weights
		
		self.T = T
		self.clear()
