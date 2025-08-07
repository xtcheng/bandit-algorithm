import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.moduleUsers.linUCB import LinUCB
from algorithms.modular.selectionModules.generalUcbModule import GeneralUCBModule
from algorithms.modular.adaptionModules.nullAdaptionModule import NullAdaptionModule
import numpy as np

class GeneralUCB(LinUCB):
	def __init__(self,T,num_arm, num_features, alpha, link_function):
		self.selection_module = GeneralUCBModule(T,num_arm,num_features,alpha, link_function)
		self.adaption_module = NullAdaptionModule(self.selection_module)
		self.T = T
		self.clear()
