# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:31:09 2023

@author: Xiaotong
"""

import numpy as np

class UCB1:
    def __init__(self,T,num_arm, xi=2):
        self.T = T
        self.num_arm = num_arm
        self.num_play = np.zeros(self.num_arm)
        self.sum_mu = np.zeros(self.num_arm)
        self.mu = np.zeros(self.num_arm)
        self.ucb = np.zeros(self.num_arm)
        self.xi = xi
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
            
            # i, which goes into the formula, changes in every turn. This means recalculations
            # must be performed for every arm, not just for the current one.
            for j in range(self.num_arm):
                if self.num_play[j] > 0:
                    self.ucb[j] = self.mu[j] + np.sqrt(self.xi*np.log(i+1)/self.num_play[j])
            
            self.sum_rgt += (br - rwd)
            self.avg_rgt[i] += self.sum_rgt/(i+1)
            self.cum_rgt[i] += self.sum_rgt
            """print("arm chosen: ", arm)
            print("reward: ", rwd)
            print("best reward: ", br)
            print("average regret at ", i, ": ", self.avg_rgt[i])
            print("cumulative regret at ", i, ": ", self.cum_rgt[i])"""
    
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
            
                
                