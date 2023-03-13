# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:53:25 2023

@author: Xiaotong
"""

from env import environment
from UCB1 import UCB1
import numpy as np
import matplotlib.pyplot as plt

num_arm = 2
T = 1000
mu = np.array([0.1,0.9])

env = environment(num_arm,mu)
UCB1 = UCB1(T,num_arm)
UCB1.run(env)
cum_rgt = UCB1.get_cum_rgt()
avg_rgt = UCB1.get_avg_rgt()

plt.figure(figsize=(6,5))
plt.plot(cum_rgt,ls='-',color = 'b',label = 'UCB1')
plt.xlabel('t (Trials)',fontsize=15) 
plt.title('Cumulative Regret') 
plt.legend(loc='upper right')   

plt.figure(figsize=(6,5))
plt.plot(avg_rgt,ls='-',color = 'b',label = 'UCB1')
plt.xlabel('t (Trials)',fontsize=15) 
plt.title('Average Regret') 
plt.legend(loc='upper right')  