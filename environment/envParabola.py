import sympy
import numpy as np

class EnvParabola:
	def __init__(self, dimensions, pos_mean=0, pos_scale=5, slope_mean=1, slope_scale=0.5, boundaries=10, stability=1):
		# dimensions: How many dimensions the input has.
		# pos_mean and pos_scale: draw the position of the local minimum from a gaussian distribution.
		# note that for a standard-parabola in one dimension, pos would be 0 and slope 1.
		# boundaries: either a list of 2-tuples or a positive number
		# stability: the propability that the cost function is not replaced by a new one in the next timestep.
		
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
		
		if isinstance(boundaries, int) or isinstance(boundaries, float):
			# if just a number is given, let all boundaries be minus that until that number.
			self.boundaries = list()
			for i in range(dimensions):
				self.boundaries.append((-boundaries, boundaries))
		else:
			self.boundaries = boundaries
		
		self.refresh()
		print(self.function)
		print()
	
	
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
		return current
	
	
	def getFunction(self):
		# Returns the current cost function as a sympy-object, supporting evaluation and diff.
		# For approaches that do assume access to a full function at every step.
		# Internally switches to the next cost function then (which may be the same as before).
		
		ret = self.function
		if np.random.rand() > self.stability:
			self.refresh()
		return ret
	
	
	def getSpace(self):
		# Returns the feasible set as list of tuples of the lower and the upper bound for each dimension.
		# Only input spaces in the form of hyperboxes are considered!
		
		return self.boundaries
	
	
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
				self.function = a*((var+b)**2)
			else:
				self.function += a*((var+b)**2)
	
	def getBest(self):
		# Returns the best possible output of the current cost function.
		# Usually 0, unless the minimum has been shifted outside the feasible space.
		
		inputs = list()
		for i in range(len(self.shifts)):
			# crop and append the optimal input for each dimension.
			temp = max(-self.shifts[i], self.boundaries[i][0])
			temp = min(temp, self.boundaries[i][1])
			inputs.append(temp)
		
		# evaluate and return at that point (without switching to the next cost function).
		return self.feedback(inputs, False)



# Tests:
f = EnvParabola(3, stability=0.5)
print(f.getBest())
for i in range(8):
	print("Best:", f.getBest(), "\t\tchosen:", f.feedback([0,0,0]))
