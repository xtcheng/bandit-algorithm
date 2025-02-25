import numpy as np
import sympy
from algorithms.gb_oco_ltc import GB_OCO_LTC
from helpers.commonTools import unit, vectorLen

# Version of GB_OCO_LTC that uses features from BGD to estimate the gradient based on single measures.
class GB_OCO_LTC_BGD(GB_OCO_LTC):
	def __init__(self, T, eta, delta):
		self.rng = np.random.default_rng()
		super().__init__(T, eta, delta)
	
	
	def run(self,env):
		env.refresh()
		
		bounds = env.getSpace()
		d = len(bounds)
		
		
		# Initialize both the starting values and the penalty weights with 0.
		var_values = np.zeros(d) # numpy may reject sympy-stuff and behave weirdly when using multiple pointers to the same thing, but seems to be safe here.
		
		# Note that we only have one lambda value this time, no matter how many constraints there are. (We actually might not even know the number of constraints.)
		# The environment only returns one single number for the constraint violation, so the only thing we can do is to weight that.
		lambda_value = 0
		
		
		minimum, maximum = env.getMinMax()
		C = maximum - minimum
		R = np.sqrt((0.5*(bounds[0][1] - bounds[0][0]))**2 * d)
		r = bounds[0][1] - bounds[0][0]
		ny = R / (C*np.sqrt(self.T)) # replaces self.eta from the original?
		delta = np.power((r * (R**2) * (d**2)) / (12*self.T), 1/3)
		alpha = np.power((3*R*d) / (2*r*np.sqrt(self.T)), 1/3)
		
		for timestep in range(1, self.T+1):
			best = env.getBest()
			
			# Add the unit vector to the input both times.
			u = unit(self.rng.normal(size=d))
			x = var_values + delta*u
			violation = env.getViolation(x)
			costs = env.feedback(x)
			
			# It should really be this simple: costs*u is an estimation for the gradient of the actual cost function, and lambda_value*(violation*u) for the gradient of the violation "function".
			var_gradient = (costs+lambda_value*violation)*u
			
			lambda_gradient = violation - self.eta*self.delta*lambda_value
			
			
			# From here, it is almost the same as before.
			
			# update x and lambda
			var_values -= ny*var_gradient
			lambda_value = max(0, lambda_value + self.eta*lambda_gradient)
			
			
			# The regret is the price we could have saved by picking the best.
			self.sum_regret += costs - best
			self.avg_regret[timestep-1] = self.sum_regret/timestep
			self.cum_regret[timestep-1] = self.sum_regret
			
			self.total_violation += violation
			self.metrics["Instantaneous Violation"][timestep-1] = violation
			self.metrics["Average Violation"][timestep-1] = self.total_violation/timestep
