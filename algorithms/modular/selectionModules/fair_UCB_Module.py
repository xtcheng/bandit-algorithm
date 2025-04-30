import numpy as np
import sys
from scipy import optimize
from helpers.nash import *
if not "../" in sys.path:
	sys.path.append('../')
from algorithms.modular.selectionModules.MO_OGDE_Module import MO_OGDE_Module


class Fair_UCB_Module(MO_OGDE_Module):
	def __init__(self, history, T, num_arm, num_objectives, delta):
		self.T = T
		self.num_arm = num_arm
		self.num_objectives = num_objectives
		self.delta = delta
		self.history = history
		self.history.register(self)
		self.fullReset()
	
	def omegaFunction(self, arm, dimension):
		# Calculates the confidence bound that is added on top of the estimation.
		log_part = np.log(4*self.num_objectives *self.num_arm *self.history.current_turn / self.delta )
		return np.sqrt(log_part / self.history.num_play[arm]) 
		#return np.sqrt( 12*(1-self.history.mu_reverse[arm][dimension])*log_part / self.history.num_play[arm])  +  12*log_part / self.history.num_play[arm]
	
	def updateMix(self):
		# Will not do it like this because the parent will choose an arm that has never been chosen, if there is any, which has the same effect as choosing the arm of the current timestep, but is safer if the history might be modified.
		"""if self.history.current_turn <= self.T:
			self.current_mix = [0]*self.num_arm
			self.current_mix[round(self.history.current_turn)-1] = 1
		else:"""
		
		if 0 in self.history.num_play: # Assumes that this will never be increased by the adaption module.
			return
		
		U = []
		for a in range(self.num_arm):
			U.append([])
			for d in range(self.num_objectives):
				U[-1].append(max(self.history.mu_reverse[a][d] - self.omegaFunction(a, d) , 0)) #
		
		# Now find the mix that maximizes the Nash function of this.
		con = [{'type':'eq', 'fun': (lambda x : sum(x)-1)}]
		result = optimize.minimize(nashReverse, args=U, x0=[1/self.num_arm]*self.num_arm, constraints=con, bounds=[(0,1)]*self.num_arm)
		self.current_mix = result.x
