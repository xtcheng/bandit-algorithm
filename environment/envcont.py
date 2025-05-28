# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 18:57:04 2025

@author: Xiaotong
"""

import numpy as np
from environment.Gaussian_noise import Gaussian_noise

def get_best_reward(items, theta):
	return np.max(np.matmul(items,theta))

def sample_items(itemset, num_items):
    items = np.zeros((num_items,itemset.shape[1]))
    for i in range(0,num_items):
        j = np.random.choice(itemset.shape[0])
        items[i] = itemset[j]
    return items

def generate_items_gaussian(M,dim,rho_sq,x_max):
    V = (1 - rho_sq) * np.eye(M) + rho_sq * np.ones((M, M))
    items = np.random.multivariate_normal(np.zeros(M), V, dim).T
    l2_norm = np.linalg.norm(items, axis=1)[:, None]
    l2_norm[l2_norm <= x_max] = x_max
    items *= x_max / l2_norm
    return items 
    
def generate_items_uniform(M,d):
    items = np.random.uniform(0,1,(M,d))
    return items

def generate_items(M, d):
    # return a ndarray of num_items * d
    x = np.random.normal(0, 1, (M, d-1))
    x = np.concatenate((np.divide(x, np.outer(np.linalg.norm(x, axis = 1), np.ones(np.shape(x)[1])))/np.sqrt(2), np.ones((M, 1))/np.sqrt(2)), axis = 1)
    return x

class environment:
    def __init__(self,K,dim,theta,sample,itemset = [],rho_sq = 0.7, x_max = 1):
        self.K = K   # number of items
        self.dim = dim
        self.theta = theta 
        self.sample = sample    # -2: fixed sample set; -1: fixed sample set with sampling; 0: gaussian distribution; 1: uniform distribution
        if self.sample < 0:
            self.itemset = itemset
        else:
            self.rho_sq = rho_sq
        self.x_max = x_max
    
    def getArmFeatures(self):
        if self.sample < 0:
            if self.sample < -1:
                self.items = sample_items(self.itemset,self.K)
            else:
                self.items = self.itemset
        else:
            if self.sample > 1.5:
                self.items = generate_items_gaussian(self.K,self.dim,self.rho_sq,self.x_max)
            else:
                self.items = generate_items_uniform(self.K,self.dim)
        return self.items
    
    def feedback(self,idx):
        x = self.items[idx,:]
        noise = Gaussian_noise(1,0,0.01,[-0.1,0.1])
        r = np.dot(self.theta,x) + noise.sample_trunc()
        # r = np.dot(self.theta[i],x) 
        br = get_best_reward(self.items,self.theta)
        return r,br
    