import numpy as np
import sympy

class GreedyProjection:
	def __init__(self, T):
		self.T = T
		self.rng = np.random.default_rng()
		self.clear()
	
	def run(self,env):
	
		# Init:
		
		env.refresh()
		
		# The boundaries and the variables will never change.
		bounds = env.getSpace()
		variables = env.getVariables()
		
		# select arbitrary x1 from F.
		x = list()
		for bound in bounds:
			# pick some valid value for each dimension.
			x.append(self.rng.uniform(bound[0], bound[1]))
		
		for timestep in range(1, self.T+1):
			best = env.getBest()
			cost_fun = env.getFunction()
			#print("Received cost function", cost_fun)
			
			# Set the learning rate as suggested in the paper.
			learning_rate = 1 / np.sqrt(timestep)
			
			# Calculate the costs of the current position.
			costs = cost_fun
			for i in range(len(variables)):
				costs = costs.subs(variables[i], x[i])
			#print("Point", x, "has costs", costs)
			
			# calculate the gradients.
			weighted_gradients = list()
			for i in range(len(variables)):
				# derive to each variable and evaluate at the selected point.
				diff = sympy.diff(cost_fun, variables[i])
				weighted_gradients.append(learning_rate * diff.subs(variables[i], x[i]))
			
			# The new input is the old input minus the weighted gradient, projected into the feasible space.
			for i in range(len(x)):
				x[i] = max(min(x[i] - weighted_gradients[i], bounds[i][1]), bounds[i][0])
			
			# The regret is the price we could have saved by picking the best.
			self.sum_regret += costs - best
			self.avg_regret[timestep-1] += self.sum_regret/timestep
			self.cum_regret[timestep-1] += self.sum_regret
	
	
	
	def get_avg_rgt(self):
		return self.avg_regret
	
	def get_cum_rgt(self):
		return self.cum_regret
	
	def clear(self):
		self.sum_regret = 0
		self.avg_regret = np.zeros(self.T)
		self.cum_regret = np.zeros(self.T)
