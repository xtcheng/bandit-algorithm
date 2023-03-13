# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:31:09 2023

@author: Xiaotong
"""

import numpy as np

class UCB1:
    def __init__(self,T,num_arm):
        self.T = T
        self.num_arm = num_arm
        self.num_play = np.zeros(self.num_arm)
        self.sum_mu = np.zeros(self.num_arm)
        self.mu = np.zeros(self.num_arm)
        self.ucb = np.zeros(self.num_arm)
        self.sum_rgt = 0
        self.avg_rgt = np.zeros(self.T)
        self.cum_rgt = np.zeros(self.T)
    
    def run(self,env):
        for i in range(0,self.T):
            if i < self.num_arm:
                arm = i
            else:
                arm = np.argmax(self.ucb)
            rwd, br = env.feedback(arm)
            self.sum_mu[arm] += rwd
            self.num_play[arm] += 1
            self.mu[arm] = self.sum_mu[arm]/self.num_play[arm]
            self.ucb[arm] = self.mu[arm] + np.sqrt(2*np.log(i)/self.num_play[arm])
            self.sum_rgt += (br - rwd)
            self.avg_rgt[i] += self.sum_rgt/(i+1)
            self.cum_rgt[i] += self.sum_rgt
    
    def get_avg_rgt(self):
        return self.avg_rgt
    
    def get_cum_rgt(self):
        return self.cum_rgt
    
    def clear(self):
        self.num_play = np.zeros(self.num_arm)
        self.sum_mu = np.zeros(self.num_arm)
        self.mu = np.zeros(self.num_arm)
        self.ucb = np.zeros(self.num_arm)
        self.sum_rgt = 0
        self.avg_rgt = np.zeros(self.T)
        self.cum_rgt = np.zeros(self.T)
            
                
                