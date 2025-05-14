# -*- coding: utf-8 -*-
"""
Created on Tue May  6 13:13:47 2025

@author: Xiaotong
"""

import sys
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.moduleUsers.basicMultiObjective import BasicMultiObjective
from algorithms.modular.selectionModules.pareto_UCB_Module import Pareto_UCB_Module
from algorithms.modular.adaptionModules.nullAdaptionModule import NullAdaptionModule
from algorithms.modular.adaptionModules.discountModule import DiscountModule
from algorithms.modular.historyContainerMO import HistoryContainerMO
from helpers.gini import gini

class DParetoUCB(BasicMultiObjective):
	def __init__(self,T,num_arm, num_objectives, alpha, gini_weights, gamma):
		self.historyContainer = HistoryContainerMO(num_arm, num_objectives)
		self.selection_module = Pareto_UCB_Module(self.historyContainer, T, num_arm, num_objectives, alpha, gini_weights)
		self.adaption_module = DiscountModule(self.historyContainer, num_arm, gamma)
		self.weights = gini_weights
		
		self.T = T
		self.clear()
