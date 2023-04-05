import numpy as np

class UCB1N:
    def __init__(self, T, num_arm, c):
        self.T = T
        self.num_arm = num_arm
        self.num_play = np.zeros(self.num_arm)
        self.sum_reward = np.zeros(self.num_arm)
        self.avg_reward = np.zeros(self.num_arm)
        self.ucb = np.zeros(self.num_arm)
        self.sum_regret = 0
        self.avg_regret = np.zeros(self.T)
        self.cum_regret = np.zeros(self.T)
        self.c = c
    
    def run(self, env):
        for i in range(self.T):
            if i < self.num_arm:
                arm = i
            else:
                ucb_values = self.avg_reward + self.c * np.sqrt(np.log(i) / self.num_play)
                upper_bound = np.inf if self.num_play[ucb_values == 0].sum() > 0 else self.c * np.sqrt(np.log(i))
                ucb_values += upper_bound * np.sqrt(np.log(i) / (self.num_play + 1))
                arm = np.argmax(ucb_values)
            
            reward, br = env.feedback(arm)
            self.sum_reward[arm] += reward
            self.num_play[arm] += 1
            self.avg_reward[arm] = self.sum_reward[arm] / self.num_play[arm]
            self.ucb[arm] = self.avg_reward[arm] + np.sqrt((self.c * np.log(i)) / (self.num_play[arm] + 1))
            self.sum_regret += (br - reward)
            self.avg_regret[i] = self.sum_regret / (i+1)
            self.cum_regret[i] = self.sum_regret

    def get_avg_rgt(self):
        return self.avg_regret
    
    def get_cum_rgt(self):
        return self.cum_regret
    
    def clear(self):
        self.num_play = np.zeros(self.num_arm)
        self.sum_reward = np.zeros(self.num_arm)
        self.avg_reward = np.zeros(self.num_arm)
        self.ucb = np.zeros(self.num_arm)
        self.sum_regret = 0
        self.avg_regret = np.zeros(self.T)
        self.cum_regret = np.zeros(self.T)
