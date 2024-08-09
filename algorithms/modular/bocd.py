import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.abstractMAB import AbstractMAB
from algorithms.modular.ucbForcedExploreModule import UCBForcedExploreModule
from algorithms.modular.bocdModule import BOCDModule

class BOCD(AbstractMAB):
	def __init__(self,T,num_arm, alpha):
		self.selection_module = UCBForcedExploreModule(T,num_arm,2,alpha)
		self.adaption_module = BOCDModule(self.selection_module, num_arm, T)
		super().__init__(T)
