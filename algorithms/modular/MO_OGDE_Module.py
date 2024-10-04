import numpy as np
import sys
from cvxopt import *
import math
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.abstractSelectionModule import AbstractSelectionModule

class MO_OGDE_Module(AbstractSelectionModule):
	def __init__(self,T,num_arm, num_objectives, delta, gini_weights):
		self.T = T
		self.num_arm = num_arm
		self.num_objectives = num_objectives
		self.delta = delta
		self.current_mix = np.array([1/num_arm]*num_arm)
		self.gini_weights = gini_weights
		self.fullReset
	
	def suggestArm(self):
		# Select each arm once before performing the meaningful operations.
		if self.current_turn <= self.num_arm:
			return self.num_arm-1
		else:
			sum_mix = 0
			for m in self.current_mix:
				sum_mix += m
			# Select a random arm according to the distribution
			return np.random.choice(range(self.num_arm), p=(self.current_mix))
	
	def thisHappened(self, arm, reward):
		# To update the information after a pull. It's still called reward, but it are actually costs this time.
		self.sum_mu[arm] += reward
		self.num_play[arm] += 1
		self.current_turn += 1
		
		for i in range(self.num_objectives):
			self.mu[arm][i] = self.sum_mu[arm][i] / self.num_play[arm]
		self.sorted_mu[arm] = sorted(self.mu[arm], reverse=True)
		
		# Compute the gradient that is assumed for gradient decent.
		gradient = np.zeros(self.num_arm)
		for i in range(self.num_arm):
			for j in range(self.num_objectives):
				gradient[i] += self.sorted_mu[i][j] * self.gini_weights[j]
			# Weight by the probability of that arm being picked because we want to evaluate the gradient at the current position.
			gradient[i] *= self.current_mix[i]
		#print(gradient)
		
		# Perform gradient decent.
		learning_rate = (np.sqrt(2) / (1 - (1/np.sqrt(self.num_arm))) ) * np.sqrt(np.log(2/self.delta) / self.current_turn)
		for i in range(self.num_arm):
			self.current_mix[i] -= learning_rate * gradient[i]
		
		# Project back into the feasible set.
		self.current_mix = self.change_A(self.current_mix, self.num_arm, self.current_turn)
	
	def fullReset(self):
		# Reset everything to the start.
		self.sum_mu = np.zeros((self.num_arm, self.num_objectives))
		self.mu = np.zeros((self.num_arm, self.num_objectives))
		self.sorted_mu = np.zeros((self.num_arm, self.num_objectives))
		self.num_play = [0]*self.num_arm
		self.relative_start = [1]*self.num_arm # But first disregard resetting single arms.
		self.current_turn = 1
		
	def change_A(self,a,K,t):
		# Taken almost directly from https://github.com/zhacheny/Optimization-based-on-GNI-Index-For-multi-objective-bandits/blob/master/Codes/LearningML.py
		# TODO: Can the projection be performed in a simpler way? If not, at least mute the output or prevent it from going to stdout.
		# TODO: Does not seem to work as intended, but throws no errors.
		
		beta = (math.sqrt(2.0)*math.sqrt(math.log(2/self.delta)/(t+K)))/(1-1/math.sqrt(K))
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
		return np.array(sol['x']).flatten()
	
