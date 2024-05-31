
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:30:21 2023
@author: Xiaotong
"""

#import sys
#import Gaussian_noise as gn

class env_stochastic:
    def __init__(self,num_arm,mu, noise):
        self.num_arm = num_arm
        self.mu = mu
        self.noise = noise
    
    def feedback(self,arm):
        rwd = self.mu[arm] + self.noise.sample_trunc()
        br  =  max(self.mu)
        return rwd, br