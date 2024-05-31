import numpy as np

class UCB1N:
    def __init__(self,T,num_arm):
        self.T = T
        self.num_arm = num_arm
        self.num_play = np.zeros(self.num_arm)
        self.sum_mu = np.zeros(self.num_arm)
        self.sum_rwd_sqr = np.zeros(self.num_arm)
        self.mu = np.zeros(self.num_arm)
        self.ucb = np.zeros(self.num_arm)
        self.sum_rgt = 0
        self.avg_rgt = np.zeros(self.T)
        self.cum_rgt = np.zeros(self.T)
    
    def run(self,env):
        for n in range(1, self.T+1):
            if n <= self.num_arm:
                arm = n-1
            else:
                max_ucb = float('-inf')
                for j in range(self.num_arm):
                    if self.num_play[j] < 8*np.log(n):
                        arm = j
                        break
                    else:
                        ucb = self.mu[j] + np.sqrt((16*(self.sum_rwd_sqr[j] 
                                    - self.num_play[j]*(self.mu[j]**2))/(self.num_play[j]-1))*(np.log(n-1)/self.num_play[j]))
                        if ucb > max_ucb:
                            max_ucb = ucb
                            arm = j
                        
            rwd, br = env.feedback(arm)
            self.sum_mu[arm] += rwd
            self.sum_rwd_sqr[arm] += rwd**2
            self.num_play[arm] += 1
            self.mu[arm] = self.sum_mu[arm]/self.num_play[arm]
            self.ucb[arm] = self.mu[arm] + np.sqrt((16*(self.sum_rwd_sqr[arm] - self.num_play[arm]*(self.mu[arm]**2))/(self.num_play[arm]-1))*(np.log(n-1)/self.num_play[arm]))
            self.sum_rgt += (br - rwd)
            self.avg_rgt[n-1] = self.sum_rgt/n
            self.cum_rgt[n-1] = self.sum_rgt
            

    
    def get_avg_rgt(self):
        return self.avg_rgt
    
    def get_cum_rgt(self):
        return self.cum_rgt
    
    def clear(self):
        self.num_play = np.zeros(self.num_arm)
        self.sum_mu = np.zeros(self.num_arm)
        self.sum_rwd_sqr = np.zeros(self.num_arm)
        self.mu = np.zeros(self.num_arm)
        self.ucb = np.zeros(self.num_arm)
        self.sum_rgt = 0
        self.avg_rgt = np.zeros(self.T)
        self.cum_rgt = np.zeros(self.T)

