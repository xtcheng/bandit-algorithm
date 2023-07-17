import numpy as np
import matplotlib.pyplot as plt
from mucb import MUCB
from env import environment
from Gaussian_noise import Gaussian_noise

num_agents = 4
num_arms = 6
num_trials = 100000
ucb1 = MUCB(num_agents, num_trials, num_arms)
mu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8]  

env = environment(num_arms, mu)

ucb1.run(env)

#print(f"Average Regret: {ucb1.get_avg_rgt()}")
#print(f"Cumulative Regret: {ucb1.get_cum_rgt()}")

plt.figure(figsize=(10, 5))
#for agent in range(num_agents):
#    plt.plot(range(num_trials), ucb1.avg_rgt[agent], label=f'Agent {agent+1}')
plt.plot(range(num_trials), np.sum(ucb1.avg_rgt,axis = 0))
plt.title('Average Regret Over Time')
plt.xlabel('Time Step')
plt.ylabel('Average Regret')
plt.legend()
plt.show()

# Plot cumulative regret
plt.figure(figsize=(10, 5))
"""for agent in range(num_agents):
    plt.plot(range(num_trials), ucb1.cum_rgt[agent], label=f'Agent {agent+1}')"""
plt.plot(range(num_trials), np.sum(ucb1.cum_rgt,axis = 0 ))
plt.title('Cumulative Regret Over Time')
plt.xlabel('Time Step')
plt.ylabel('Cumulative Regret')
plt.legend()
plt.show()

