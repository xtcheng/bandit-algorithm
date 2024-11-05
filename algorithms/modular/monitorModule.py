import numpy as np
from collections import deque

class MonitorModule:
	def __init__(self, selection_module, num_arm, w, b, dimensions=1):
		self.selection_module = selection_module
		self.num_arm = num_arm
		self.len_window = w
		self.detection_threshold = b
		self.dimensions = dimensions
		self.fullReset()
	
	
	def changeDetection(self, arm, dimension):
		# Assume that sum_rewards0 and sum_rewards already contain the correct results for the current point in time. This is true the first time this function is called for any arm (after start or an reset) because the number of pulls for that arm will equal len_window, so sum_rewards0 is the sum of rewards [1, len_window/2] and sum_rewards0 is the sum of [len_window/2 +1, len_window]. So just use the sums unmodified.
		result = np.abs(self.sums_rewards0[arm][dimension] - self.sums_rewards1[arm][dimension]) > self.detection_threshold
		
		# Now make sure the invariant will also be true next time for this arm.
		
		# The oldest element in the left part of the current window will be outside the scope of the next window, so undo its addition and pop it so next time the 2nd-oldest one will be removed and so on.
		window = self.past_rewards[arm][dimension]
		self.sums_rewards0[arm][dimension] -= window[0].popleft()
		
		# The oldest element in the right part of the current window will be inside the left part of the next window.
		temp = window[1].popleft()
		self.sums_rewards1[arm][dimension] -= temp
		self.sums_rewards0[arm][dimension] += temp
		window[0].append(temp)
		
		return result
	
	
	
	def thisHappened(self, arm, reward, timestep):
		self.num_play[arm] += 1
		
		if self.dimensions == 1:
			# Compatibility
			reward = [reward]
		
		# Safe the rewards in the first deque until it is "full". Then add rewards only to the second list. They will eventually transist to the first list via the pop-logic in changeDetection.
		if self.num_play[arm] <= self.len_window/2:
			for dimension in range(self.dimensions):
				self.past_rewards[arm][dimension][0].append(reward[dimension])
				self.sums_rewards0[arm][dimension] += reward[dimension]
		else:
			for dimension in range(self.dimensions):
				self.past_rewards[arm][dimension][1].append(reward[dimension])
				self.sums_rewards1[arm][dimension] += reward[dimension]
		
		# Unless both parts of the window are not "full" yet, perform change detection.
		if self.num_play[arm] >= self.len_window:
			change_detected = False
			for dimension in range(self.dimensions):
				change_detected = change_detected or self.changeDetection(arm, dimension)
			if change_detected:
				print("Change detected at timestep", timestep)
				self.selection_module.fullReset()
				self.fullReset()
	
	
	
	
	def fullReset(self):
		# Needed for changeDetection: The sum of the left and the right part of the window, per arm and dimension.
		self.sums_rewards0 = [0]*self.num_arm
		self.sums_rewards1 = [0]*self.num_arm
		for i in range(self.num_arm):
			self.sums_rewards0[i] = [0]*self.dimensions
			self.sums_rewards1[i] = [0]*self.dimensions
		self.num_play = [0]*self.num_arm
		
		# Past rewards. List that for each arm and dimension contains a 2-tuple of deques of the history.
		self.past_rewards = list()
		for i in range(self.num_arm):
			self.past_rewards.append(list())
			for j in range(self.dimensions):
				# Important: Use deque so that, unlike with list, not only append but also popleft runs in O(1) and inplace.
				self.past_rewards[-1].append((deque(), deque()))
