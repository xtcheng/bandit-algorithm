import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.basicMultiObjective import BasicMultiObjective
from algorithms.modular.MO_OGDE_Module import MO_OGDE_Module
from algorithms.modular.slidingWindowModule import SlidingWindowModule
from algorithms.modular.gini import gini

class SlidingWindowMO(BasicMultiObjective):
	def __init__(self,T,num_arm, num_objectives, delta, gini_weights, window_len):
		self.selection_module = MO_OGDE_Module(T,num_arm, num_objectives, delta, gini_weights)
		self.adaption_module = SlidingWindowModule(self.selection_module, num_arm, window_len)
		self.weights = gini_weights
		
		self.T = T
		self.clear()
