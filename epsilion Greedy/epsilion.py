import numpy as np

class EpsilonGreedy:
    def __init__(self, T, num_arm,c,d):
        self.T = T
        self.num_arm = num_arm
        self.c = c
        self.d = d
        self.num_play = np.zeros(num_arm)
        self.sum_mu = np.zeros(num_arm)
        self.mu = np.zeros(num_arm)
        self.cum_rgt = np.zeros(T)
        self.avg_rgt = np.zeros(T)
    
    def run(self, env):      
        for i in range(self.T):
            n = sum(self.num_play)
            e = min(1,self.c/(self.d**2*n))
            if np.random.random() < e:
                arm = np.random.choice(self.num_arm)
            else:
                arm = np.argmax(self.mu)
            rwd, br = env.feedback(arm)
            self.sum_mu[arm] += rwd
            self.num_play[arm] += 1
            self.mu[arm] = self.sum_mu[arm] / self.num_play[arm]
            self.cum_rgt[i] = self.cum_rgt[i-1] + (br - rwd)
            self.avg_rgt[i] = self.cum_rgt[i] / (i+1)
    
    def get_avg_rgt(self):
        return self.avg_rgt
    
    def get_cum_rgt(self):
        return self.cum_rgt
    
    def clear(self):
        self.num_play = np.zeros(self.num_arm)
        self.sum_mu = np.zeros(self.num_arm)
        self.mu = np.zeros(self.num_arm)
        self.cum_rgt = np.zeros(self.T)
        self.avg_rgt = np.zeros(self.T)
