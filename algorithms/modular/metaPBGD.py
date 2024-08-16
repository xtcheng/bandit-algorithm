import numpy as np
import sympy
from algorithms.commonTools import unit
from algorithms.modular.bgdExpert import BGD_Expert

class MetaPBGD:
	# Following the paper https://jmlr.org/papers/v22/20-763.html
	
	def __init__(self, T, len_experts=100, ):
		self.T = T
		self.len_experts = len_experts
		self.rng = np.random.default_rng()
		self.clear()
	
	def run(self,env):
	
		# Most of the init is like before:
		bounds = env.getSpace()
		d = len(bounds)
		
		L = env.getLipschitz()
		
		# The upper bound for the costs is needed later. This algorithm seems to assume the minimum is zero.
		C = env.getMinMax()[1]
		
		# Big R is the radius of the ball that contains the feasible set.
		# It is the distance from the center of the cube to a corner.
		R = np.sqrt((0.5*(bounds[0][1] - bounds[0][0]))**2 * d)
		
		# Small r is the radius of the ball that is contained in the feasible set.
		# It is the distance from the center of the cube to a face.
		r = bounds[0][1] - bounds[0][0]
		
		# Calculate the parameters for the general case.
		# TODO: Use the parameters that exploid the Lipschitzness.
		delta = np.power((r * (R**2) * (d**2)) / (12*self.T), 1/3)
		alpha = np.power((3*R*d) / (2*r*np.sqrt(self.T)), 1/3)
		
		large_grad = d*C / delta
		epsilon = np.sqrt(1/ (2*self.T*(large_grad**2)*(R**2)) )
		
		# Initialize the experts.
		experts = []
		for i in range(1, self.len_experts+1):
			rate = 2**(i-1) * np.sqrt((7*(R**3)) / (2*d*C*L)) * (self.T**(-3/4))
			experts.append(BGD_Expert(alpha, rate, d, R))
		
		
		# Initalize the weights for the experts.
		weights = []
		for i in range(1, self.len_experts+1):
			weights.append( ((self.len_experts+1) / self.len_experts) / (i*(i+1)) )
		
		
		for timestep in range(1, self.T+1):
			best = env.getBest()
			
			# random unit vector
			s = unit(self.rng.normal(size=d))
			
			# Ask each expert for their opinion. Calculate the weighted sum and save the individual opinions for later.
			meta_y = np.zeros(d)
			ops = []
			for i in range(self.len_experts):
				ops.append(experts[i].suggest())
				meta_y += weights[i] * ops[i]
			
			x = meta_y + delta*s
			cost = env.feedback(x)
			grad = (d/delta) * cost * s
			
			# Calculate the surrogate loss for each expert, modify it according to the formula (line 9 in Algorithm 2) and multiply with weight.
			# Also keep track of the total sum of these.
			losses = []
			loss_sum = 0
			for i in range(self.len_experts):
				# Did not work properly when the dot product was zero, so check wether it is even required to calculate a surrogate loss.
				if np.array_equal(ops[i], meta_y):
					raw_loss = cost
				else:
					raw_loss = np.dot(grad, ops[i] - meta_y)
				losses.append(weights[i] * np.exp(-epsilon * raw_loss))
				loss_sum += raw_loss
			
			# Update the expert weights.
			for i in range(self.len_experts):
				weights[i] = losses[i] / loss_sum
			
			# Finally, inform all experts about the estimated gradient so they can think about their next suggestion.
			for expert in experts:
				expert.setGradient(grad)
			# So each expert believes their point had that gradient, even though the point that was actually picked may be far away from the point the expert suggested. Is this really how it is supposed to work?
			
			# The regret is the price we could have saved by picking the best.
			self.sum_regret += cost - best
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