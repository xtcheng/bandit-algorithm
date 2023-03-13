# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:30:21 2023

@author: Xiaotong
"""

import Gaussian_noise as gn

class environment:
    def __init__(self,num_arm,mu):
        self.num_arm = num_arm
        self.mu = mu
    
    def feedback(self,arm):
        noise = gn.Gaussian_noise(1,0,0.01,[-0.1,0.1])
        rwd = self.mu[arm] + noise.sample_trunc()
        br  =  max(self.mu)
        return rwd, br
        
