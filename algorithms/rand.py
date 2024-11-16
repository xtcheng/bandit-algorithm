# main.py
import numpy as np
import matplotlib.pyplot as plt
from env import environment
from MAMAB_SI import MAMAB_SI

# Parameters from the paper
T = 50000  
num_agents = 6
num_arms = 10  
mu = [0.4, 0.5, 0.5, 0.6, 0.7, 0.7, 0.8, 0.9, 0.92, 0.95]  


observation_matrix_case1 = [
    [0, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0]

]
observation_matrix_case2 = [
    [0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 0],
    [1, 0, 1, 0, 0, 0]
]

# Sociabilities from the paper for both the cases 
sociabilities_case1 = [0.60, 0.5, 0.05, 0.10, 0.2, 0.90]
#sociabilities_case2 = [0.5, 0.1, 0.6, 0.4]


env = environment(num_arms, mu)


algorithm_case1 = MAMAB_SI(T, num_agents, num_arms, sociabilities_case1, observation_matrix_case1)
algorithm_case1.run(env)


#algorithm_case2 = MAMAB_SI(T, num_agents, num_arms, sociabilities_case2, observation_matrix_case2)
#algorithm_case2.run(env)

# Get cumulative regret for both cases and sum over all agents
cum_regret_case1_total = algorithm_case1.get_cum_rgt()
avg_regret_case1_total = algorithm_case1.get_avg_rgt()
'''
cum_regret_case2_total = algorithm_case2.get_cum_rgt()
avg_regret_case2_total = algorithm_case2.get_avg_rgt()
'''
# Plotting
fig, axs = plt.subplots( figsize=(12, 5))

# Plot a regret for cases 
axs.plot(cum_regret_case1_total,color='red', label='cum_regret_case1')
#axs.plot(cum_regret_case2_total,color='blue', label='cum_regret_case2')
axs.set_title('Cumulative Regret of MAMAB_SI Algorithm')
axs.set_xlabel('Trials')
axs.set_ylabel('Total Cumulative Regret')
axs.legend()
axs.grid(True)

plt.tight_layout()
plt.show()

#plot average 
fig, axs = plt.subplots( figsize=(12, 5))

# Plot a regret for cases 
axs.plot(avg_regret_case1_total,color='red', label='avg_regret_case1')
#axs.plot(cum_regret_case2_total,color='blue', label='cum_regret_case2')
axs.set_title('Average Regret of MAMAB_SI Algorithm')
axs.set_xlabel('Trials')
axs.set_ylabel('Total Cumulative Regret')
axs.legend()
axs.grid(True)

plt.tight_layout()
plt.show()


# Plotting
fig, axs = plt.subplots( figsize=(10, 12))

# Plot cumulative regret for each agent for Case 1
for agent in range(num_agents):
    axs.plot(algorithm_case1.cum_rgt[agent], label=f'Agent {agent+1}')
axs.set_title('Cumulative Regret of MAMAB_SI Algorithm (Case 1)')
axs.set_xlabel('Trials')
axs.set_ylabel('Cumulative Regret')
axs.legend()
axs.grid(True)
'''
# Plot cumulative regret for each agent for Case 2
for agent in range(num_agents):
    axs[1].plot(algorithm_case2.cum_rgt[agent], label=f'Agent {agent+1}')
axs[1].set_title('Cumulative Regret of MAMAB_SI Algorithm (Case 2)')
axs[1].set_xlabel('Trials')
axs[1].set_ylabel('Cumulative Regret')
axs[1].legend()
axs[1].grid(True)
'''
plt.tight_layout()
plt.show()
