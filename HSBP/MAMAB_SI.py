
import numpy as np

class MAMAB_SI:
    def __init__(self, T, num_agents, num_arm, sociabilities, observation_matrix):
        self.T = T
        self.num_agents = num_agents
        self.num_arm = num_arm
        self.sociabilities = sociabilities
        self.observation_matrix = observation_matrix
        
        self.num_play = np.zeros((self.num_agents, self.num_arm))
        self.sum_mu = np.zeros((self.num_agents, self.num_arm))
        self.mu = np.zeros((self.num_agents, self.num_arm))
        self.ucb = np.zeros((self.num_agents, self.num_arm))
        self.sum_rgt = np.zeros(self.num_agents)
        self.avg_rgt = np.zeros((self.num_agents, self.T))
        self.cum_rgt = np.zeros((self.num_agents, self.T))
    
    def run(self, env):
        for i in range(self.T):
            for agent in range(self.num_agents):
                if i < self.num_arm:
                    arm = i
                else:
                    arm = np.argmax(self.ucb[agent]) #uses the highest UCB value... when there is tie... 
                
                rwd, br = env.feedback(arm)
                self.update(agent, arm, rwd, br, i)
                
                # Stochastic observation of neighbors
                for neighbor in range(self.num_agents):
                    if self.observation_matrix[agent][neighbor] and np.random.rand() < self.sociabilities[agent]:
                        self.update2(agent, arm, rwd, br, i)
    # Q! As stated from the paper that agent observes neighbours based on the sociability value, however the value is assessed randomly, but can also be deterministic by adding a rule, rounding up to one, this could imply anything above 0.5 would be highly socialable and below they dont interact with neighbours, which could induce ambiguity... 
    def update(self, agent, arm, rwd, br, i):
        self.sum_mu[agent][arm] += rwd
        self.num_play[agent][arm] += 1
        self.mu[agent][arm] = self.sum_mu[agent][arm] / self.num_play[agent][arm]
        self.ucb[agent][arm] = self.mu[agent][arm] + np.sqrt(2 * np.log(i + 1) / self.num_play[agent][arm])
        self.sum_rgt[agent] += (br - rwd)
        self.avg_rgt[agent][i] = self.sum_rgt[agent] / (i + 1)
        self.cum_rgt[agent][i] = self.sum_rgt[agent]
    
    def update2(self, agent, arm, rwd, br, i):
        self.sum_mu[agent][arm] += rwd
        self.num_play[agent][arm] += 1
        self.mu[agent][arm] = self.sum_mu[agent][arm] / self.num_play[agent][arm]
        self.ucb[agent][arm] = self.mu[agent][arm] + np.sqrt(2 * np.log(i + 1) / self.num_play[agent][arm])

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
