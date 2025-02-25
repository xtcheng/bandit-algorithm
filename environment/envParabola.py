import sympy
import numpy as np
import math
from scipy import optimize

class EnvParabola:
	def __init__(self, dimensions, pos_mean=0, pos_scale=5, slope_mean=1, slope_scale=0.5, boundaries=10, stability=1, fixed_breakpoints=False):
		# dimensions: How many dimensions the input has.
		# pos_mean and pos_scale: draw the position of the local minimum from a gaussian distribution.
		# note that for a standard-parabola in one dimension, pos would be 0 and slope 1.
		# boundaries: either a list of 2-tuples or a positive number
		# stability: the propability that the cost function is NOT replaced by a new one in the next timestep.
		
		variable_string = "x0"
		for i in range(1, dimensions):
			variable_string += " x"+str(i)
		self.variables = sympy.symbols(variable_string)
		
		self.pos_mean = pos_mean
		self.pos_scale = pos_scale
		if slope_mean <= 0:
			print("Warning: slope_mean should be positive to avoid an excessive amount of redraws.")
		self.slope_mean = slope_mean
		self.slope_scale = slope_scale
		self.stability = stability
		self.fixed_breakpoints = fixed_breakpoints
		# Position of the breakpoints may be fixed instead of random to better compare things.
		
		if stability < 1 and fixed_breakpoints:
			self.breakpoint_target = 1 / (1-stability)
			self.breakpoint_status = 0
		
		if isinstance(boundaries, int) or isinstance(boundaries, float):
			# if just a number is given, let all boundaries be minus that until that number.
			self.boundaries = list()
			for i in range(dimensions):
				self.boundaries.append((-boundaries, boundaries))
		else:
			self.boundaries = boundaries
		
		self.constraints = []
		
		#self.refresh() call this in the strategy to avoid having no random init due to deepcopy!
	
	def addConstraint(self, constraint):
		self.constraints.append(constraint)
	
	def getVariables(self):
		# Returns the list of variables that are the input of every cost function.
		# This implicitly includes the input dimension.
		
		return self.variables
	
	
	def feedback(self, inputs, switch=True):
		# Returns the value of the current cost function at the position of input.
		# For approaches that do not assume access to a full function at every step.
		# Internally switches to the next cost function then (which may be the same as before).
		
		current = self.function
		for i in range(len(self.variables)):
			current = current.subs(self.variables[i], inputs[i])
		
		if switch and np.random.rand() > self.stability:
			self.refresh()
		return float(current)
	
	def getViolation(self, inputs):
		# Returns the violation of the constraint that was violated most.
		# Note that non-violated constraints will have no effect; even overly well kept contraints will not improve the result.
		result = 0
		for constraint in self.constraints:
			violation = constraint
			for i in range(len(self.variables)):
				violation = violation.subs(self.variables[i], inputs[i])
			#print(inputs, "evaluate to", violation)
			result = max(result, float(violation))
		return result
	
	def isViolating(self, inputs):
		# Returns true if the inputs violate at least one of the constraints.
		return (self.getViolation(inputs) > 0)
	
	def getFunction(self):
		# Returns the current cost function as a sympy-object, supporting evaluation and diff.
		# For approaches that do assume access to a full function at every step.
		# Internally switches to the next cost function then (which may be the same as before).
		
		ret = self.function
		if not self.fixed_breakpoints:
			if np.random.rand() > self.stability:
				self.refresh()
		else:
			self.breakpoint_status += 1-self.stability
			if self.breakpoint_status >= self.breakpoint_target:
				self.breakpoint_status = 0
				self.refresh()
		return ret
	
	
	def getSpace(self):
		# Returns the feasible set as list of tuples of the lower and the upper bound for each dimension.
		# Only input spaces in the form of hyperboxes are considered!
		
		return self.boundaries
	
	def getConstraints(self):
		return self.constraints
	
	def getMinMax(self):
		# Return the smallest and the greatest possible return of any cost function.
		# The smallest is 0 and the greatest is at the edge of a function that has its minimum on the other edge.
		# TODO: Consider the case where the minimum is not inside the feasible set.
		
		slope = self.slope_mean + 2*self.slope_scale # TODO: guarantee that the slope will never be greater.
		maximum = 0
		for bound in self.boundaries:
			maximum += (bound[1]-bound[0])**2 * slope
		return 0, maximum
	
	def getLipschitz(self):
		# Return the L that satisfies the Lipschitz criteria.
		
		# We deal with parabolas, which have simpler shape than a general L-Lipschitz-function. In fact, it should be sufficient to set L to the maximum possible gradient, because the area of this is where the difference between two outputs would show the greatest deviation from the according inputs.
		
		a = self.slope_mean + 2*self.slope_scale
		
		# Use the dimensions with the largest intervall
		selected_dimension = (0,0)
		for dimension in self.boundaries:
			if dimension[1]-dimension[0] > selected_dimension[1]-selected_dimension[0]:
				selected_dimension = dimension
		b = -selected_dimension[0]
		
		var = sympy.symbols("test_x")
		test_fun = a * a*((var+b)**2)
		diff = sympy.diff(test_fun, var)
		return float(diff.subs(var, selected_dimension[1]))
		
	
	def refresh(self):
		# Internally used to get a new cost function.
		
		# Build the sub-expressions, e.g. 3*(x+1).
		self.function = 0
		self.shifts = list()
		for var in self.variables:
			a = 0
			while a <= 0:
				a = np.random.normal(self.slope_mean, self.slope_scale)
			b = np.random.normal(self.pos_mean, self.pos_scale)
			self.shifts.append(b)
			if self.function==0:
				self.function = a*((var-b)**2)
			else:
				self.function += a*((var-b)**2)
		self.best = math.inf
	
	def evalConstraint(self, inputs, index):
		result = self.constraints[index]
		for i in range(len(self.variables)):
			result = result.subs(self.variables[i], inputs[i])
		return float(result)
		
	
	def getBest(self):
		# Returns the best possible output of the current cost function.
		# Usually 0, unless the minimum has been shifted outside the feasible space.
		# Ask this BEFORE the cost function / costs!
		
		if self.best == math.inf:
			# If the function did not change, we have already cached the optimal result.
			
			cons = []
			for i in range(len(self.constraints)):
				cons.append({'type':'ineq', 'fun': (lambda x : -self.evalConstraint(x, i))})
			result = optimize.minimize(self.feedback, args=False, x0=[0]*len(self.variables), constraints=cons, bounds=self.boundaries)
			#print(result)
			self.best=result.fun
			# Note that this can be off by about 10^(-17) and may lead to very slightly wrong regret results!
		return self.best


"""
# Tests:
f = EnvParabola(3, stability=0.5)
print(f.getBest())
for i in range(8):
	print("Best:", f.getBest(), "\t\tchosen:", f.feedback([0,0,0]))
"""
