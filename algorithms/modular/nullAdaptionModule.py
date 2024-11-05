# No adaption - breakpoints will not be recognized or even assumed.
class NullAdaptionModule:
	def __init__(self, selection_module):
		self.selection_module = selection_module
	
	def thisHappened(self, arm, reward, t):
		pass
	
	def fullReset(self):
		pass
