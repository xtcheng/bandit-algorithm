import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.moduleUsers.linUCB import LinUCB
from algorithms.modular.selectionModules.CW_OFUL_Module import CW_OFUL_Module
from algorithms.modular.adaptionModules.nullAdaptionModule import NullAdaptionModule
import numpy as np

class CW_OFUL(LinUCB):
	def __init__(self,T,num_arm, num_features, lmda, beta, alpha):
		self.selection_module = CW_OFUL_Module(T, num_arm, num_features, lmda, beta, alpha) # The only difference.
		self.adaption_module = NullAdaptionModule(self.selection_module)
		self.T = T
		self.clear()
