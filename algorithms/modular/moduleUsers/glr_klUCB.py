import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.abstractMAB import AbstractMAB
from algorithms.modular.ucbForcedExploreModule import UCBForcedExploreModule
from algorithms.modular.glrModule import GLRModule

class GLR_klUCB(AbstractMAB):
	def __init__(self,T,num_arm, alpha, delta, global_restart, lazyness=1):
		self.selection_module = UCBForcedExploreModule(T,num_arm,2,alpha)# TODO: Implement klUCB instead.
		self.adaption_module = GLRModule(self.selection_module, num_arm, delta, global_restart, lazyness)
		super().__init__(T)
