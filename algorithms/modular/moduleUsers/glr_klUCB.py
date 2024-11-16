import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.moduleUsers.abstractMAB import AbstractMAB
from algorithms.modular.selectionModules.ucbForcedExploreModule import UCBForcedExploreModule
from algorithms.modular.adaptionModules.glrModule import GLRModule

class GLR_klUCB(AbstractMAB):
	def __init__(self,T,num_arm, alpha, delta, global_restart, lazyness=1):
		self.selection_module = UCBForcedExploreModule(T,num_arm,2,alpha)# TODO: Implement klUCB instead.
		self.adaption_module = GLRModule(self.selection_module, num_arm, delta, global_restart, lazyness)
		super().__init__(T)
