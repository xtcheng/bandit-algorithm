import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.moduleUsers.abstractMAB import AbstractMAB
from algorithms.modular.selectionModules.ucbForcedExploreModule import UCBForcedExploreModule
from algorithms.modular.adaptionModules.monitorModule import MonitorModule

class MonitoredUCB(AbstractMAB):
	def __init__(self,T,num_arm, w, b, gamma):
		self.selection_module = UCBForcedExploreModule(T,num_arm,0.5,gamma)
		self.adaption_module = MonitorModule(self.selection_module, num_arm, w, b)
		super().__init__(T)
