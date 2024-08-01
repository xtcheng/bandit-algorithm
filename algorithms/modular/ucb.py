import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.abstractMAB import AbstractMAB
from algorithms.modular.ucbModule import UCBModule
from algorithms.modular.nullAdaptionModule import NullAdaptionModule

class ModularUCB(AbstractMAB):
	def __init__(self,T,num_arm, xi):
		self.xi = xi
		self.selection_module = UCBModule(T,num_arm,xi)
		self.adaption_module = NullAdaptionModule(self.selection_module)
		super().__init__(T,num_arm)
