import numpy as np

class CoopUCB2Selective:
    def __init__(self, T, num_agents, num_arms, observation_matrix, gamma, sigma_g, eta):
        self.T = T  
        self.num_agents = num_agents  
        self.num_arms = num_arms  
        self.gamma = gamma  
        self.eta = eta  
        self.sigma_g = sigma_g  
        
        # Initialize variables
        self.n = np.zeros((self.num_agents, self.num_arms))  
        self.s = np.zeros((self.num_agents, self.num_arms))  
        self.mu = np.zeros((self.num_agents, self.num_arms))  
        self.Q = np.zeros((self.num_agents, self.num_arms))  
        self.W = np.zeros((self.num_agents, self.num_arms))  
        

        L = np.diag(np.sum(observation_matrix, axis=1)) - observation_matrix
        self.P = np.eye(self.num_agents) - self.eta * L


        self.cumulative_regret = np.zeros((self.num_agents, self.T))
        self.average_regret = np.zeros((self.num_agents, self.T))
        self.sum_rgt = np.zeros(self.num_agents)

    def f(self, t):

        return np.sqrt(np.log(t))

    def G(self, eta):

        return 1 - ((eta**2) / 16)

    def update_ucb(self, agent, t):

        for arm in range(self.num_arms):
            M = self.num_arms
            self.Q[agent][arm] = self.mu[agent][arm] + self.sigma_g * np.sqrt(
                (2 * self.gamma / self.G(self.eta)) * ((self.n[agent][arm] + self.f(t-1)) / (M * self.n[agent][arm])) * np.log(t-1) / self.n[agent][arm]
            )

    def update_lcb(self, agent, t):

        for arm in range(self.num_arms):
            self.W[agent][arm] = self.mu[agent][arm] - self.sigma_g * np.sqrt(
                (2 * self.gamma / self.G(self.eta)) * ((self.n[agent][arm] + self.f(t-1)) / (self.num_arms * self.n[agent][arm])) * np.log(t-1) / self.n[agent][arm]
            )

    def run(self, env):

        for t in range(1, self.T + 1):
            selected_arms = np.full(self.num_agents, -1)  
            
            if t <= self.num_arms:
                for agent in range(self.num_agents):
                    arm = (t-1 + agent) % self.num_arms
                    selected_arms[agent] = arm
            else:
                for agent in range(self.num_agents):

                    self.update_ucb(agent, t)

                    sorted_indices = np.argsort(-self.Q[agent])

                    k_best_arms = sorted_indices[:agent+1]


                    self.update_lcb(agent, t)

                    worst_arm_index = np.argmin([self.W[agent][arm] for arm in k_best_arms])
                    arm = k_best_arms[worst_arm_index]


                    if arm not in selected_arms:
                        selected_arms[agent] = arm


            for agent, arm in enumerate(selected_arms):
                if arm == -1:  
                    continue

                reward, _ = env.feedback(arm)

                self.n[agent][arm] += 1
                self.s[agent][arm] += reward
                self.mu[agent][arm] = self.s[agent][arm] / self.n[agent][arm]


                self.n[:, arm] = np.dot(self.P, self.n[:, arm])
                self.s[:, arm] = np.dot(self.P, self.s[:, arm])


                best_reward = max([self.mu[agent][arm] for arm in range(self.num_arms)])
                regret = best_reward - reward


                self.sum_rgt[agent] += regret
                self.cumulative_regret[agent][t-1] = self.sum_rgt[agent]
                self.average_regret[agent][t-1] = self.cumulative_regret[agent][t-1] / t

    def get_avg_rgt(self):

        return np.mean(self.average_regret, axis=0)

    def get_cum_rgt(self):

        return np.sum(self.cumulative_regret, axis=0)

    def clear(self):

        self.n = np.zeros((self.num_agents, self.num_arms))
        self.s = np.zeros((self.num_agents, self.num_arms))
        self.mu = np.zeros((self.num_agents, self.num_arms))
        self.Q = np.zeros((self.num_agents, self.num_arms))
        self.W = np.zeros((self.num_agents, self.num_arms))
        self.cumulative_regret = np.zeros((self.num_agents, self.T))
        self.average_regret = np.zeros((self.num_agents, self.T))
        self.sum_rgt = np.zeros(self.num_agents)
