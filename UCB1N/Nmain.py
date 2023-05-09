from envN import environment
from UCB1N import UCB1N
import numpy as np
import matplotlib.pyplot as plt

num_arm = 2
T = 100
mu = np.array([0.1, 0.9])


env = environment(num_arm, mu)
UCB1N = UCB1N(T, num_arm, c=1)
UCB1N.run(env)
cum_rgt = UCB1N.get_cum_rgt()
avg_rgt = UCB1N.get_avg_rgt()

plt.figure(figsize=(6,5))
plt.plot(cum_rgt,ls='-',color = 'b',label = 'UCB1-N (c=1)')
plt.xlabel('t (Trials)',fontsize=15) 
plt.title('Cumulative Regret') 
plt.legend(loc='upper right')   
plt.show()

plt.figure(figsize=(6,5))
plt.plot(avg_rgt,ls='-',color = 'b',label = 'UCB1-N (c=1)')
plt.xlabel('t (Trials)',fontsize=15) 
plt.title('Average Regret') 
plt.legend(loc='upper right')  
plt.show()