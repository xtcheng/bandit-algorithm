import numpy as np

class MUCB:
    def __init__(self, num_agents, T, num_arm):
        self.T = T
        self.num_agents = num_agents
        self.num_arm = num_arm
        self.num_play = np.zeros((self.num_agents, self.num_arm))
        self.sum_mu = np.zeros((self.num_agents, self.num_arm))
        self.mu = np.zeros((self.num_agents, self.num_arm))
        self.ucb = np.zeros((self.num_agents, self.num_arm))
        self.sum_rgt = np.zeros(self.num_agents)
        self.avg_rgt = np.zeros((self.num_agents, self.T))
        self.cum_rgt = np.zeros((self.num_agents, self.T))
    
    @staticmethod
    def hungarian(matrix):
        n, m = matrix.shape
        u = matrix.max() - matrix
        result = []
        for _ in range(n):
            min_val = np.inf
            min_idx = (-1, -1)
            for i in range(n):
                for j in range(m):
                    if u[i, j] < min_val:
                        min_val = u[i, j]
                        min_idx = (i, j)
            u[min_idx] = np.inf
            result.append(min_idx)
        return result
    
    def run(self, env):
        for i in range(self.T):
            if i < self.num_arm:
                for agent in range(self.num_agents):
                    arm = i 
                    rwd, br = env.feedback(arm)
                    self.update(agent, arm, rwd, br, i)
            else:
                assignments = self.hungarian(self.ucb)

                for agent, arm in assignments:
                    rwd, br = env.feedback(arm)
                    self.update(agent, arm, rwd, br, i)
    
    def update(self, agent, arm, rwd, br, i):
        self.sum_mu[agent][arm] += rwd
        self.num_play[agent][arm] += 1
        self.mu[agent][arm] = self.sum_mu[agent][arm] / self.num_play[agent][arm]
        self.ucb[agent][arm] = self.mu[agent][arm] + np.sqrt(2 * np.log(i + 1) / self.num_play[agent][arm])
        self.sum_rgt[agent] += (br - rwd)
        self.avg_rgt[agent][i] = self.sum_rgt[agent] / (i + 1)
        self.cum_rgt[agent][i] = self.sum_rgt[agent]
    
    def get_avg_rgt(self):
        return np.sum(self.avg_rgt, axis=0)
    
    def get_cum_rgt(self):
        return np.sum(self.cum_rgt, axis=0)
    
    def clear(self):
        self.num_play = np.zeros((self.num_agents, self.num_arm))
        self.sum_mu = np.zeros((self.num_agents, self.num_arm))
        self.mu = np.zeros((self.num_agents, self.num_arm))
        self.ucb = np.zeros((self.num_agents, self.num_arm))
        self.sum_rgt = np.zeros(self.num_agents)
        self.avg_rgt = np.zeros((self.num_agents, self.T))
        self.cum_rgt = np.zeros((self.num_agents, self.T))
