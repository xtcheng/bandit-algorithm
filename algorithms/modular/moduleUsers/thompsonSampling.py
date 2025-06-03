import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.moduleUsers.abstractMAB import AbstractMAB
from algorithms.modular.selectionModules.thompsonModule import ThompsonModule
from algorithms.modular.adaptionModules.nullAdaptionModule import NullAdaptionModule

class ThompsonSampling(AbstractMAB):
	def __init__(self,T,num_arm):
		self.selection_module = ThompsonModule(T,num_arm)
		self.adaption_module = NullAdaptionModule(self.selection_module)
		super().__init__(T)
