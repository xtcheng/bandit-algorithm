
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
    
    def feedbackMult(self,arms):
        # Draw the selected arms and output 0 for all others.
        rewards = list()
        for i in range(self.num_arm):
            if i in arms:
                rewards.append(self.mu[i] + self.noise.sample_trunc())
            else:
                rewards.append(0)
        # The best possible reward is by pulling the k best arms.
        br = 0
        for expectation in sorted(self.mu, reverse=True)[:len(arms)]:
            br += expectation
        return rewards, br
