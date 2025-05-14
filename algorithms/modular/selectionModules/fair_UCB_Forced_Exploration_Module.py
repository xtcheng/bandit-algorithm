# -*- coding: utf-8 -*-
"""
Created on Sat May  3 16:27:46 2025

@author: Xiaotong
"""

import numpy as np
import sys
from scipy import optimize
import os
from cvxopt import *
import math
from helpers.gini import gini
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.selectionModules.MO_OGDE_Forced_Exploration_Module import MO_OGDE_Forced_Exploration_Module
from algorithms.modular.selectionModules.abstractSelectionModule import AbstractSelectionModule

def ggi(x, mu):
	weights = []
	for i in range(mu.shape[1]):
		weights.append(1/(2**i))
        
	avg_cost = x @ mu
	#print(costs)
	costs = sorted(avg_cost, reverse=True)
	index = sum(w * c for w, c in zip(weights, costs))
	
	return index

class Fair_UCB_Forced_Exploration_Module(MO_OGDE_Forced_Exploration_Module):
	def __init__(self, history, T, num_arm, num_objectives, delta, alpha):
		self.T = T
		self.num_arm = num_arm
		self.num_objectives = num_objectives
		self.delta = delta
		self.alpha = alpha
		self.history = history
		self.history.register(self)
		self.fullReset()
 	
	def omegaFunction(self, arm, dimension):
		# Calculates the confidence bound that is added on top of the estimation.history.current_turn
		log_part = np.log(4*self.num_objectives *self.num_arm *self.history.current_turn)
		return np.sqrt(log_part / 2*self.history.num_play[arm]) 
		#return np.sqrt( 12*(1-self.history.mu_reverse[arm][dimension])*log_part / self.history.num_play[arm])  +  12*log_part / self.history.num_play[arm]
 	
	def updateMix(self):
		# Will not do it like this because the parent will choose an arm that has never been chosen, if there is any, which has the same effect as choosing the arm of the current timestep, but is safer if the history might be modified.
		"""if self.history.current_turn <= self.T:
 			self.current_mix = [0]*self.num_arm
 			self.current_mix[round(self.history.current_turn)-1] = 1
		else:"""
		
		if 0 in self.history.num_play: # Assumes that this will never be increased by the adaption module.
			return
		
		lcb = np.zeros_like(self.history.mu)
		for a in range(self.num_arm):
			for d in range(self.num_objectives):
				lcb[a][d] += (self.history.mu[a][d] - self.omegaFunction(a,d))
		
		# Now find the mix that maximizes the Nash function of this.
		con = [{'type':'eq', 'fun': (lambda x : sum(x)-1)}]
		result = optimize.minimize(ggi, 
                             args=(self.history.mu,), 
                             x0=np.array([1/self.num_arm]*self.num_arm), 
                             method='SLSQP', 
                             constraints=con, 
                             bounds=[(0,1)]*self.num_arm,
                             options={'ftol': 1e-12,     # objective function tolerance
                                 'maxiter': 1000,   # max number of iterations
                                 'disp': False       # print optimization output
                                 })
		self.current_mix = result.x
		self.current_mix /= np.sum(self.current_mix)
# 		if self.history.current_turn % 100 == 0:
# 			print(lcb)
