from heapq import *

class Tuple:
	def __init__(self, name, count):
		self.name = name
		self.count = count
	
	# enable comparison (flip < and > because we want to sort descending)
	def __le__(self, other):
		return self.count >= other.count
	def __lt__(self, other):
		return self.count > other.count
	
	# For printing
	def __repr__(self):
		return "\'" + self.name + "\': " + str(self.count)

def dict2heap(d):
	# python's heap uses a list and is not a new class.
	heap = []
	for key in d:
		heappush(heap, Tuple(name=key, count=d[key]))
	return heap


if __name__ == "__main__":
	k = 10
	
	mapping = dict() # which meaning is at which position
	
	# How many times these show up
	user_counter = dict()
	track_counter = dict()
	track_counter_user = dict()
	track_counter_time = (dict(), dict(), dict())
	
	print("Parsing file...")
	with open("Last.fm_data.csv", "r", encoding="utf-8") as file:
		
		line0 = file.readline().split(',')
		for i in range(len(line0)):
			mapping[line0[i].strip()] = i
		
		for line in file:
			# parse line and clean up
			proto_elements = line.strip().split(',')
			elements = []
			pending = "" # The elements that wait for merge because they actually belong to the same column
			# The tricky part: Because the data uses ',' to seperate the columns and ',' shows up as part of some titles, these titles are protected with '"'. But this can also be part of the title, and English opening and closing '"' are identical...
			for element in proto_elements:
				if element[0] != '\"' and element[-1] != '\"':
					if pending == "":
						elements.append(element)
					else:
						pending += element
				elif element[-1] == '\"':
					# So it ends with '"', but can we really close it or is this part of the name?
					if (pending.count('\"') + element.count('\"')) % 2 == 0:
						elements.append((pending + element)[1:-1])
						pending = ""
					else:
						pending += element
				else:
					pending += element
			
			# Tricky part done, check if the interpretation looks sane.
			if len(elements) != len(mapping):
				print(elements, "is broken, has", len(elements), "entries instead of", len(mapping))
				print(line)
				continue
			
			username = elements[mapping["Username"]]
			if not username in user_counter:
				user_counter[username] = 0
				track_counter_user[username] = dict() # then it's not there yet either.
			user_counter[username] += 1
			
			track = elements[mapping["Track"]]
			if not track in track_counter:
				track_counter[track] = 0
			track_counter[track] += 1
			
			time = elements[mapping["Time"]]
			hour = int(time.split(':')[0])
			# day, evening, night.
			if hour >= 6 and hour < 18:
				dayphase = 0
			elif hour >= 18 and hour < 22:
				dayphase = 1
			else:
				dayphase = 2
			if not track in track_counter_time[dayphase]:
				track_counter_time[dayphase][track] = 0
			track_counter_time[dayphase][track] += 1
			
			if not track in track_counter_user[username]:
				track_counter_user[username][track] = 0
			track_counter_user[username][track] += 1
	print("Reading complete.")
	
	# We want to find the k largest of n unsorted elements, where k << n (at least for the tracks).
	# -> use priority queue (O(k*log(n))) instead of full sort (O(n*log(n))): https://docs.python.org/3/library/heapq.html
	user_ranking = dict2heap(user_counter)
	track_ranking = dict2heap(track_counter)
	track_ranking_time = (dict2heap(track_counter_time[0]), dict2heap(track_counter_time[1]), dict2heap(track_counter_time[2]))
	track_ranking_user = dict()
	for user in track_counter_user:
		track_ranking_user[user] = dict2heap(track_counter_user[user])
	
	biggest_users = []
	print("\nThe", k, "most active users:\n")
	for i in range(min(k, len(user_ranking))):
		biggest_users.append(heappop(user_ranking))
		print(biggest_users[-1])
	
	print("\n\nThe", k, "most popular tracks:\n")
	for i in range(min(k, len(track_ranking))):
		print(heappop(track_ranking))
	
	print("\n\nThe", k, "most popular tracks at day:\n")
	for i in range(min(k, len(track_ranking_time[0]))):
		print(heappop(track_ranking_time[0]))
	
	print("\n\nThe", k, "most popular tracks at evening:\n")
	for i in range(min(k, len(track_ranking_time[1]))):
		print(heappop(track_ranking_time[1]))
	
	print("\n\nThe", k, "most popular tracks at night:\n")
	for i in range(min(k, len(track_ranking_time[2]))):
		print(heappop(track_ranking_time[2]))
	
	for user in biggest_users:
		user = user.name # only need the name here.
		print("\n\nThe", k, "most popular tracks of "+user+":\n")
		for i in range(min(k, len(track_ranking_user[user]))):
			print(heappop(track_ranking_user[user]))
