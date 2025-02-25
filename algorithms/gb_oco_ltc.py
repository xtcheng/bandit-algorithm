import numpy as np
import sympy

class GB_OCO_LTC:
	def __init__(self, T, eta, delta):
		self.T = T
		self.eta = eta
		self.delta = delta
		self.clear()
	
	def run(self,env):
	
		# Init:
		
		env.refresh()
		
		bounds = env.getSpace()
		
		# Expect a list of sympy functions, each being one constraint that is satified when <= 0.
		constraints = env.getConstraints()
		
		# The variables that are in play.
		vars = env.getVariables()
		d = len(vars)
		
		# The constraints will not change, so we can calculate the derivatives already here.
		constraint_dervs = []
		for constraint in constraints:
			constraint_dervs.append([])
			# we need one derivative per variable.
			for var in vars:
				constraint_dervs[-1].append(sympy.diff(constraint, var))
		
		
		# Initialize both the starting values and the penalty weights with 0.
		var_values = [0]*d
		lambda_values = [0]*len(constraints)
		
		
		for timestep in range(1, self.T+1):
			best = env.getBest()
			has_violated = env.isViolating(var_values)
			violation = env.getViolation(var_values)
			cost_fun = env.getFunction()
			
			# Calculate the costs of the current position.
			costs = cost_fun
			for i in range(d):
				costs = costs.subs(vars[i], var_values[i])
			
			# We are looking for the actual gradient plus penalty.
			var_gradient = [0]*d
			for i in range(d):
				diff = sympy.diff(cost_fun, vars[i])
				actual = diff.subs(vars[i], var_values[i])
				penalty = 0
				for j in range(len(constraint_dervs)):
					penalty += lambda_values[j] * constraint_dervs[j][i].subs(vars[i], var_values[i])
				var_gradient[i] = actual + penalty
			
			lambda_gradient = [0]*len(constraints)
			for j in range(len(constraint_dervs)):
				violation_per_constraint = constraints[j]
				for i in range(d):
					violation_per_constraint = violation_per_constraint.subs(vars[i], var_values[i])
				lambda_gradient[j] = violation_per_constraint - self.eta*self.delta*lambda_values[j]
			
			# update x and lambda
			for i in range(d): # Saver than adding numpy arrays
				var_values[i] -= self.eta*var_gradient[i]
				# And projection into cube B.
				var_values[i] = max(bounds[i][0], var_values[i])
				var_values[i] = min(bounds[i][1], var_values[i])
			for j in range(len(constraint_dervs)):
				lambda_values[j] = max(0, lambda_values[j] + self.eta*lambda_gradient[j])
			#print("\nT =", timestep, "\nlambda gradient:", lambda_gradient, "\nlambda values:", lambda_values)
			
			
			# The regret is the price we could have saved by picking the best.
			self.sum_regret += costs - best
			self.avg_regret[timestep-1] = self.sum_regret/timestep
			self.cum_regret[timestep-1] = self.sum_regret
			
			self.total_violation += violation
			self.metrics["Instantaneous Violation"][timestep-1] = violation
			self.metrics["Average Violation"][timestep-1] = self.total_violation/timestep
	
	
	
	def get_avg_rgt(self):
		return self.avg_regret
	
	def get_cum_rgt(self):
		return self.cum_regret
	
	def getMetric(self, key):
		return self.metrics[key]
	
	def listMetrics(self):
		return {"Instantaneous Violation", "Average Violation"}
	
	def clear(self):
		self.metrics = dict()
		self.sum_regret = 0
		self.avg_regret = np.zeros(self.T)
		self.cum_regret = np.zeros(self.T)
		self.metrics["Instantaneous Violation"] = np.zeros(self.T)
		self.total_violation = 0
		self.metrics["Average Violation"] = np.zeros(self.T)
