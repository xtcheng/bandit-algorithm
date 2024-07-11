
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:30:21 2023
@author: Xiaotong
"""

import numpy as np

class env_stochastic:
    def __init__(self,num_arm,mu, noise, stability=1):
        self.num_arm = num_arm
        self.mu = mu
        self.noise = noise
        self.stability = stability
        self.rng = np.random.default_rng()
    
    def feedback(self,arm):
        rwd = self.mu[arm] + self.noise.sample_trunc()
        br  =  max(self.mu)
        self.shuffleMaybe()
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
        self.shuffleMaybe()
        return rewards, br
    
    def shuffleMaybe(self):
        # New: An arms expectation value may be redrawn with propability 1-stability. This happens independently for any arm at any timestep, even if the arm was not chosen. Therefore, the propability that ALL arms are unaffected by this is stability powered by the the number of arms, so set stability high enough when there are many arms to avoid creating a much less stable environment than expected. The new value of an arm is drawn uniformly from [0,1). 
        for i in range(self.num_arm):
            if self.rng.uniform() >= self.stability:
                self.mu[i] = self.rng.uniform()
