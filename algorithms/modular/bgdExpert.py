import numpy as np
import sympy
from algorithms.commonTools import *

# The experts for the meta function. They hardly do anything on their own, they just suggest inputs and receive estimated gradients.
class BGD_Expert:
	def __init__(self, alpha, eta, d, chi):# No need for T because the loop is external.
		self.alpha = alpha
		self.eta = eta # Learning rate. The letter was not ny after all.
		self.d = d # Number of dimensions
		self.chi = chi # radius of the ball that contains the feasible set
		self.rng = np.random.default_rng()
		self.clear()
	
	
	def suggest(self):
		# Simply return the desired input for the current timestep. (The combination with a unit vector happens externally.)
		return self.y
	
	
	def setGradient(self, gradient):
		# Be informed about the (estimated) gradient of the last suggestion and use it to calculate the next suggestion.
		
		# Substract the gradient, diminished according to the step size eta, from the last input.
		y_raw = self.y - self.eta*gradient
		
		# And project it into the sphere.
		if vectorLen(y_raw) > (1-self.alpha) * self.chi:
			self.y = (1-self.alpha) * self.chi * unit(y_raw)
		else:
			self.y = y_raw
	
	
	def clear(self):
		self.y = np.zeros(self.d)
