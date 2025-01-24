from heapq import *
import numpy as np

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
	
	mapping = dict() # which meaning is at which position
	
	# How many times these show up
	user_counter = dict()
	track_counter = dict()
	track_counter_user = dict()
	track_counter_time = (dict(), dict(), dict())
	total_calls = 0
	
	print("Parsing file...")
	
	with open("Last.fm_data.csv", "r", encoding="utf-8") as file:
		line0 = file.readline().split(',')
		for i in range(len(line0)):
			mapping[line0[i].strip()] = i
	
	# Avoid messing with the details and let numpy figure out the rest. Not trivial because ',' is used to seperate items and '"' to delimit some, but both may be part of the items as well!
	data = np.loadtxt("Last.fm_data.csv", encoding="utf-8", delimiter=',', skiprows=1, dtype=str, quotechar='"', comments=None)
	for elements in data:
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
		
		total_calls += 1
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
	print("\nThe", k_users, "most active users:\n")
	for i in range(min(k_users, len(user_ranking))):
		biggest_users.append(heappop(user_ranking))
		print(biggest_users[-1])
	
	popular_tracks = []
	# We will only consider these tracks.
	print("\n\nThe", k_tracks, "most popular tracks:\n")
	for i in range(min(k_tracks, len(track_ranking))):
		popular_tracks.append(heappop(track_ranking))
		print(popular_tracks[-1])
	
	print("\n\nThe", k_tracks, "most popular tracks at day:\n")
	for i in range(min(k_tracks, len(track_ranking_time[0]))):
		print(heappop(track_ranking_time[0]))
	
	print("\n\nThe", k_tracks, "most popular tracks at evening:\n")
	for i in range(min(k_tracks, len(track_ranking_time[1]))):
		print(heappop(track_ranking_time[1]))
	
	print("\n\nThe", k_tracks, "most popular tracks at night:\n")
	for i in range(min(k_tracks, len(track_ranking_time[2]))):
		print(heappop(track_ranking_time[2]))
	
	for user in biggest_users:
		user = user.name # only need the name here.
		print("\n\nThe", k_tracks, "most popular tracks of "+user+":\n")
		for i in range(min(k_tracks, len(track_ranking_user[user]))):
			print(heappop(track_ranking_user[user]))
	
	mu = np.zeros((len(popular_tracks), len(biggest_users)))
	# a track is an arm and the arm's dimensions are how much each user likes it.
	armstr = ""
	dimstr = ""
	for i_track, track in enumerate(popular_tracks):
		armstr += track.name + ", "
		for i_user, user in enumerate(biggest_users):
			if i_track == 0:
				dimstr += user.name + ", "
			if track.name in track_counter_user[user.name]:
				# if the user has listened to this track at least once, define their response for this track by the share this track had in the user's total history, including(!) tracks that are not overally popular. You might instead only want to consider popular tracks that the user has listened to, which will usually lead to a higher fraction and make each given dimension among all arms sum up to ~1.
				rating = track_counter_user[user.name][track.name] / user_counter[user.name]
			else:
				# If there is no data for this track and user, make something up. Will use the overal popularity of the track among all users here. Alternatives would be to pick a random number from some interval or use 0.
				rating = track_counter[track.name] / total_calls
			# Turn rating (which is always between 0 and 1) into costs
			mu[i_track][i_user] = 1 - rating
	np.savetxt("fm_banditified.csv", mu, delimiter=',', header="Arms(tracks): ["+armstr[:-2]+"]\ndimensions(users): ["+dimstr[:-2]+"]", encoding="utf-8")
