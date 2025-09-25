import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.moduleUsers.linUCB import LinUCB
from algorithms.modular.selectionModules.linThompsonModule import LinThompsonModule
from algorithms.modular.adaptionModules.nullAdaptionModule import NullAdaptionModule
import numpy as np

class LinThompson(LinUCB):
	def __init__(self,T,num_arm, num_features, alpha, delta, sigma):
		self.selection_module = LinThompsonModule(T, num_arm, num_features, alpha, delta, sigma) # The only difference.
		self.adaption_module = NullAdaptionModule(self.selection_module)
		self.T = T
		self.clear()
