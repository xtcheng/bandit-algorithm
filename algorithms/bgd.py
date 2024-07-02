import numpy as np
import sympy

class BGD:
	def __init__(self, T):
		self.n = T
		self.rng = np.random.default_rng()
		self.clear()
	
	def unit(self, vector):
		temp_len = 0
		for a in vector:
			temp_len += float(a**2) # cast to float because it may be a sympy-object. And numpy doesn't like that.
		return (1/np.sqrt(temp_len)) * vector
	
	def vectorLen(self, vector):
		temp_len = 0
		for a in vector:
			temp_len += float(a**2)
		return temp_len
	
	def run(self,env):
	
		# Init:
		
		# The boundaries and the variables will never change.
		bounds = env.getSpace()
		d = len(bounds)
		
		# Suppose there is a C: output is in [-C, C]. Then C is maximum output value
		# minus the minimum. Map the output into that space using an offset.
		# For example, if the actual output range is from 0 to 10,
		# then we would interpret  0 as -5,  1 as -4 ... 10 as 5.
		minimum, maximum = env.getMinMax()
		C = maximum - minimum
		
		# TODO: Calculate an input offset in case the cube is not centered at 0.
		input_offsets = np.zeros(d)
		
		# the nonexistent previous modified output is set to zero.
		y = np.zeros(d)
		
		# Big R is the radius of the ball that contains the feasible set.
		# It is the distance from the center of the cube to a corner.
		R = np.sqrt((0.5*(bounds[0][1] - bounds[0][0]))**2 * d)
		
		# Small r is the radius of the ball that is contained in the feasible set.
		# It is the distance from the center of the cube to a face.
		r = bounds[0][1] - bounds[0][0]
		
		# Calculate the parameters for the general case.
		ny = R / (C*np.sqrt(self.n))
		delta = np.power((r * (R**2) * (d**2)) / (12*self.n), 1/3)
		alpha = np.power((3*R*d) / (2*r*np.sqrt(self.n)), 1/3)
		
		for timestep in range(1, self.n+1):
			best = env.getBest()
			
			# random unit vector
			u = self.unit(self.rng.random(size=d))
			
			# Calculate the next input.
			x = y + delta*u
			#print("input:", x)
			
			# Get the costs of the current position.
			costs = env.feedback(x + input_offsets)
			
			# Calculate the next y.
			y_raw = y - ny*costs*u
			
			# And project it into the sphere
			if self.vectorLen(y_raw) > (1-alpha) * R:
				y = (1-alpha) * R * self.unit(y_raw)
			else:
				y = y_raw
			#print(y_raw, y)
			#print(self.vectorLen(y_raw), self.vectorLen(y))
			
			
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
		self.avg_regret = np.zeros(self.n)
		self.cum_regret = np.zeros(self.n)