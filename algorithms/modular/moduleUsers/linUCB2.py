import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.moduleUsers.linUCB import LinUCB
from algorithms.modular.selectionModules.linUcbModule2 import LinUCBModule2
from algorithms.modular.adaptionModules.nullAdaptionModule import NullAdaptionModule

class LinUCB2(LinUCB):
	def __init__(self,T,num_arm, num_features, alpha):
		self.selection_module = LinUCBModule2(T,num_arm,num_features,alpha)
		self.adaption_module = NullAdaptionModule(self.selection_module)
		super().__init__(T,num_arm, num_features, alpha)

