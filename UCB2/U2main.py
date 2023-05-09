from env import environment
from UCB2 import UCB2
import numpy as np
import matplotlib.pyplot as plt

num_arms = 2
T = 100
mu = np.array([0.1, 0.2])
alpha = 5

env = environment(num_arms, mu)
UCB2 = UCB2(T,num_arms, alpha)
UCB2.run(env)
cum_rgt = UCB2.get_cum_rgt()
avg_rgt = UCB2.get_avg_rgt()

plt.figure(figsize=(6,5))
plt.plot(cum_rgt,ls='-',color = 'b',label = 'UCB2')
plt.xlabel('t (Trials)',fontsize=15) 
plt.title('Cumulative Regret') 
plt.legend(loc='upper right')   


plt.figure(figsize=(6,5))
plt.plot(avg_rgt,ls='-',color = 'b',label = 'UCB2')
plt.xlabel('t (Trials)',fontsize=15) 
plt.title('Average Regret') 
plt.legend(loc='upper right')  
plt.show()