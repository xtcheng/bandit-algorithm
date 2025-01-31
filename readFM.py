from heapq import *
import numpy as np
from sys import argv

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
	k_tracks = 10 # how many of the n tracks to consider
	k_users = 10 # how many of the n users to consider
	
	assert len(argv) == 2, "Please provide the filename"
	infilename = argv[1]
	
	use_correction = False
	# This is wether or not to apply a modifier that is the number of times a user has listened to
	# the most k popular tracks divided by the number of times they have listed to their own k most
	# popular tracks. If the latter is much higher, this means that even the general popular tracks
	# they have listened to most will not satisfy them very much. The user will have low rewards in all
	# dimension and will be widely ignored by most strategies as there isn't much we can do for them using
	# our selection anyway.
	# Vice versa, if the user's preferences largely overlap with the general popular track or are
	# widely spread, this factor will not do much.
	
	# Avoiding magic numbers
	TOTAL = 0
	DAY = 1
	EVENING = 2
	NIGHT = 3
	
	mapping = dict() # which meaning is at which position
	
	# How many times these show up
	user_counter = dict() # total only because we cannot swap out the meanings of the dimensions in the middle.
	track_counter = (dict(), dict(), dict(), dict())
	track_counter_user = (dict(), dict(), dict(), dict())
	total_calls = [0]*4
	
	print("Parsing file...")
	with open(infilename, "r", encoding="utf-8") as file:
		
		line0 = file.readline().split(',')
		for i in range(len(line0)):
			mapping[line0[i].strip()] = i
		
		for line in file:
			# parse line and clean up
			proto_elements = line.strip().split(',')
			elements = []
			pending = "" # The elements that wait for merge because they actually belong to the same column
			# The tricky part: Because the data uses ',' to seperate the columns and ',' shows up as part of some 
			# titles, these titles are protected with '"'. But this can also be part of the title, and English 
			# opening and closing '"' are identical...
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
				for i in range(len(track_counter_user)):
					track_counter_user[i][username] = dict()# then it's not there yet either.
			user_counter[username] += 1
			
			track = elements[mapping["Track"]]
			if not track in track_counter[TOTAL]:
				track_counter[TOTAL][track] = 0
			track_counter[TOTAL][track] += 1
			
			time = elements[mapping["Time"]]
			hour = int(time.split(':')[0])
			# day, evening, night.
			if hour >= 6 and hour < 18:
				dayphase = DAY
			elif hour >= 18 and hour < 22:
				dayphase = EVENING
			else:
				dayphase = NIGHT
			if not track in track_counter[dayphase]:
				track_counter[dayphase][track] = 0
			track_counter[dayphase][track] += 1
			
			if not track in track_counter_user[TOTAL][username]:
				track_counter_user[TOTAL][username][track] = 0
			if not track in track_counter_user[dayphase][username]:
				track_counter_user[dayphase][username][track] = 0
			track_counter_user[TOTAL][username][track] += 1
			track_counter_user[dayphase][username][track] += 1
			
			total_calls[TOTAL] += 1
			total_calls[dayphase] += 1
	print("Reading complete.")
	
	# We want to find the k largest of n unsorted elements, where k << n (at least for the tracks).
	# -> use priority queue (O(k*log(n))) instead of full sort (O(n*log(n))): https://docs.python.org/3/library/heapq.html
	
	user_ranking = dict2heap(user_counter)
	track_ranking = (dict2heap(track_counter[0]), dict2heap(track_counter[1]), dict2heap(track_counter[2]), dict2heap(track_counter[3]))
	track_ranking_user = (dict(), dict(), dict(), dict())
	
	for user in track_counter_user[TOTAL]:
		for i in range(len(track_ranking_user)):
			track_ranking_user[i][user] = dict2heap(track_counter_user[i][user])
	
	# This is to count how many times the user has listened to their own k_tracks most popular tracks.
	# This will usually be a higher number if the user's preferences are concentrated at some few tracks.
	total_personal_calls_user = (dict(), dict(), dict(), dict())
	
	# And this is how many times the user has listened to the k_tracks most popular tracks.
	# The more the user's preferences overlap with the majority's, the closer this will be to 
	# total_personal_calls_user.
	# It will never be higher than that because the sum of an arbitrary selection of k elements from a set can 
	# never be higher than the sum of the highest k elements from the same set.
	total_popular_calls_user = (dict(), dict(), dict(), dict())
	
	biggest_users = []
	print("\nThe", k_users, "most active users:\n")
	for i in range(min(k_users, len(user_ranking))):
		biggest_users.append(heappop(user_ranking))
		print(biggest_users[-1])
	
	popular_tracks = []
	# We will only consider these tracks.
	print("\n\nThe", k_tracks, "most popular tracks:\n")
	for i in range(min(k_tracks, len(track_ranking[TOTAL]))):
		popular_tracks.append(heappop(track_ranking[TOTAL]))
		print(popular_tracks[-1])
		
	
	print("\n\nThe", k_tracks, "most popular tracks at day:\n")
	for i in range(min(k_tracks, len(track_ranking[DAY]))):
		print(heappop(track_ranking[DAY]))
	
	print("\n\nThe", k_tracks, "most popular tracks at evening:\n")
	for i in range(min(k_tracks, len(track_ranking[EVENING]))):
		print(heappop(track_ranking[EVENING]))
	
	print("\n\nThe", k_tracks, "most popular tracks at night:\n")
	for i in range(min(k_tracks, len(track_ranking[NIGHT]))):
		print(heappop(track_ranking[NIGHT]))
	
	for user in biggest_users:
		user = user.name # only need the name here.
		print("\n\nThe", k_tracks, "most popular tracks of "+user+":\n")
		for dayphase in range(len(total_personal_calls_user)):
			total_personal_calls_user[dayphase][user] = 0
			total_popular_calls_user[dayphase][user] = 0
			for i in range(min(k_tracks, len(track_ranking_user[dayphase][user]))):
				element = heappop(track_ranking_user[dayphase][user])
				total_personal_calls_user[dayphase][user] += element.count
				if dayphase == TOTAL:
					print(element)
			for track in popular_tracks:
				if track.name in track_counter_user[dayphase][user]:
					total_popular_calls_user[dayphase][user] += track_counter_user[dayphase][user][track.name]
		print("\nHas listened to their own popular tracks", total_personal_calls_user[TOTAL][user], "times.")
		print("Has listened to the overally popular tracks", total_popular_calls_user[TOTAL][user], "times.")
	
	for dayphase in range(len(track_counter)):
		mu = np.zeros((len(popular_tracks), len(biggest_users)))
		# a track is an arm and the arm's dimensions are how much each user likes it.
		armstr = ""
		dimstr = ""
		for i_track, track in enumerate(popular_tracks):
			armstr += track.name + ", "
			for i_user, user in enumerate(biggest_users):
				if use_correction:
					correction_factor = total_popular_calls_user[dayphase][user.name] / total_personal_calls_user[dayphase][user.name]
				else:
					correction_factor = 1
				if i_track == 0:
					dimstr += user.name + ", "
				if track.name in track_counter_user[dayphase][user.name]:
					# if the user has listened to this track at least once, define their response for this track by the number of times they listened to it divided by the number of times they listened to any of the most popular tracks.
					rating = track_counter_user[dayphase][user.name][track.name] / total_popular_calls_user[dayphase][user.name]
				else:
					# If there is no data for this track and user, make something up. Will use the overal popularity of the track among all users here. Alternatives would be to pick a random number from some interval or use 0.
					if track.name in track_counter[dayphase]:
						rating = track_counter[dayphase][track.name] / total_calls[dayphase]
					else:
						# if the track was never called at all by any user at the dayphase, we assume that nobody will be interested in it.
						#print(track.name, "is missing at phase", dayphase)
						rating = 0
				# Turn rating (which is always between 0 and 1) into costs and apply factor before
				mu[i_track][i_user] = 1 - rating*correction_factor
		if dayphase == TOTAL:
			filename = "fm_banditified.csv"
		elif dayphase == DAY:
			filename = "fm_banditified_day.csv"
		elif dayphase == EVENING:
			filename = "fm_banditified_evening.csv"
		else:
			filename = "fm_banditified_night.csv"
		np.savetxt(filename, mu, delimiter=',', header="Arms(tracks): ["+armstr[:-2]+"]\ndimensions(users): ["+dimstr[:-2]+"]\ncalls="+str(total_calls[dayphase]), encoding="utf-8")
