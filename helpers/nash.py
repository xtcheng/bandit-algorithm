def nash(x, ests):
	# x is how you choose to pull the arms.
	# ests are the expected rewards, or the estimations
	
	num_arms = len(ests)
	num_dimensions = len(ests[0])
	result = 1
	for d in range(num_dimensions):
		# For each agent, calculate how much we satisfy them.
		reward = 0
		for a in range(num_arms):
			reward += x[a]*ests[a][d]
		
		# and multiply those go the nash social welfare. (The version that is a simple product.
		result *= reward
	
	return result

def nashReverse(x, ests):
	# If we want to minimize.
	return -nash(x, ests)
