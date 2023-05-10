# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:54:12 2023

@author: Xiaotong
"""

from env import environment
from UCB2 import UCB2
from epsilon import EpsilonGreedy
from UCB1 import UCB1
from UCB1N import UCB1N 
import numpy as np
import matplotlib.pyplot as plt

num_arm = 5
T = 5000
mu = np.array([0.1,0.9,0.4,0.3,0.6])

env = environment(num_arm,mu)

######TO DO
''' Compare all the algorithms' performance
    run each algorithm 10 times and obtain the average regret'''
    
Trial = 10
for i in range (0,Trial):
    

    


####Plot all the regret in one figure 
plt.figure(figsize=(6,5))
#TODO
plt.xlabel('t (Trials)',fontsize=15) 
plt.title('Cumulative Regret') 
plt.legend(loc='upper right')   
plt.show()

plt.figure(figsize=(6,5))
#TODO
plt.xlabel('t (Trials)',fontsize=15) 
plt.title('Average Regret') 
plt.legend(loc='upper right')  
plt.show()
