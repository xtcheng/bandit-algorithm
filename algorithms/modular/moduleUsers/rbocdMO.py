# -*- coding: utf-8 -*-
"""
Created on Sat May  3 16:33:30 2025

@author: Xiaotong
"""

import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.moduleUsers.basicMultiObjective import BasicMultiObjective
from algorithms.modular.selectionModules.MO_OGDE_Forced_Exploration_Module import MO_OGDE_Forced_Exploration_Module
from algorithms.modular.adaptionModules.bocdModule import BOCDModule
from algorithms.modular.historyContainerMO import HistoryContainerMO
from helpers.gini import gini

class RBOCD_MO(BasicMultiObjective):
	def __init__(self,T,num_arm, num_objectives, delta, gini_weights, alpha):
		self.historyContainer = HistoryContainerMO(num_arm, num_objectives)
		self.selection_module = MO_OGDE_Forced_Exploration_Module(self.historyContainer, T, num_arm, num_objectives, delta, gini_weights, alpha)
		self.adaption_module = BOCDModule(self.historyContainer, num_arm, T, num_objectives)
		self.weights = gini_weights
		
		self.T = T
		self.clear()