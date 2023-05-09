from env import environment
from epsilion import EpsilonGreedy
import numpy as np
import matplotlib.pyplot as plt

num_arm = 2
T = 1000
mu = np.array([0.1,0.9])
c,d = 1,1

env = environment(num_arm,mu)
EG = EpsilonGreedy(T,num_arm,c,d)
EG.run(env)
cum_rgt = EG.get_cum_rgt()
avg_rgt = EG.get_avg_rgt()


plt.figure(figsize=(6,5))
plt.plot(cum_rgt,ls='-',color = 'b',label = 'Epsilon-Greedy')
plt.xlabel('t (Trials)',fontsize=15) 
plt.title('Cumulative Regret') 
plt.legend(loc='upper right')   

plt.figure(figsize=(6,5))
plt.plot(avg_rgt,ls='-',color = 'b',label = 'Epsilon-Greedy')
plt.xlabel('t (Trials)',fontsize=15) 
plt.title('Average Regret') 
plt.legend(loc='upper right')

plt.show()
