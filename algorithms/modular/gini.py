def gini(x, args):
	# x is how you choose to pull the arms.
	# args[1:] is the expected cost per arm, which are the individual costs per dimension.
	# args[0] are the weights that shall be used. Just make sure to use the same weights everywhere.
	# The optimizer expects such a parameter structure. But you can also use it to calculate the Gini index of average costs you have collected by setting all x[0] to 1 (so it does nothing) and putting the costs into args[1]. I do that for the regret analysis.
	
	num_dim = len(args[1])
	weights = args[0]
	costs = [0]*num_dim
	for k in range(len(args)-1):
		for d in range(num_dim):
			#print("Adding dimension", d, "for arm", k)
			costs[d] += x[k] * args[k+1][d]
	index = 0
	#print(costs)
	costs = sorted(costs, reverse=True)
	for i in range(len(costs)):
		index += weights[i] * costs[i]
	
	return index
