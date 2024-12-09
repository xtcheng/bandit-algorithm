import numpy as np
import sys
import os
from cvxopt import *
import math
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.selectionModules.abstractSelectionModule import AbstractSelectionModule

class MO_OGDE_Module(AbstractSelectionModule):
	def __init__(self, history, T, num_arm, num_objectives, delta, gini_weights):
		self.T = T
		self.num_arm = num_arm
		self.num_objectives = num_objectives
		self.delta = delta
		self.gini_weights = gini_weights
		self.history = history
		self.history.register(self)
		self.fullReset()
	
	def suggestArm(self):
		# Select each arm once before performing the meaningful operations.
		if 0 in self.history.num_play:
			#print("Index", self.history.num_play.index(0), "is 0!")
			return self.history.num_play.index(0)
		else:
			sum_mix = 0
			for m in self.current_mix:
				sum_mix += m
			# Select a random arm according to the distribution
			return np.random.choice(range(self.num_arm), p=(self.current_mix))
	
	def getLearningRate(self):
		return (math.sqrt(2.0)*math.sqrt(math.log(2/self.delta)/(self.history.current_turn+self.num_arm)))/(1-1/math.sqrt(self.num_arm))
	
	def updateMix(self):
		# Compute the gradient that is assumed for gradient decent.
		gradient = np.zeros(self.num_arm)
		for i in range(self.num_arm):
			for j in range(self.num_objectives):
				gradient[i] += self.history.sorted_mu[i][j] * self.gini_weights[j]
			# Weight by the probability of that arm being picked because we want to evaluate the gradient at the current position.
			gradient[i] *= self.current_mix[i]
		#print(gradient)
		
		# Perform gradient decent.
		learning_rate = self.getLearningRate()
		for i in range(self.num_arm):
			self.current_mix[i] -= learning_rate * gradient[i]
		
		# Project back into the feasible set.
		self.current_mix = self.change_A(self.current_mix, self.num_arm, self.history.current_turn)
	
	def fullReset(self):
		self.current_mix = np.array([1/self.num_arm]*self.num_arm)
	
	
	def change_A(self,a,K,t):
		# Taken almost directly from https://github.com/zhacheny/Optimization-based-on-GNI-Index-For-multi-objective-bandits/blob/master/Codes/LearningML.py
		# TODO: Can the projection be performed in a simpler way?
		
		# Send the output into space
		stdout = sys.stdout
		sys.stdout = open(os.devnull, 'w')
		
		beta = self.getLearningRate()
		beta /= K
		if(beta >1/K):
			beta = 1/K
		M_A = matrix([a[i] for i in range(0,K)],(K,1))
		# create P
		p=[]
		for i in range(0,K):
			p.append(1.0)
		ones = matrix([p[i] for i in range(0,K)],(K,1))
		P =(2.0)*spdiag(p)

		#create q
		q = -2.0*M_A
		#create G
		G = -1.0*((1.0)/(2.0))*P
		#create h
		h=-1*beta*ones
		# create A
		A =  matrix([p[i] for i in range(0,K)],(1,K))
		b = matrix(1.0)
		sol = solvers.qp(P, q, G, h, A, b)
		
		# Restore stdout
		sys.stdout = stdout
		
		return np.array(sol['x']).flatten()
	
