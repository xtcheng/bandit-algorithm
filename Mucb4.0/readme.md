# Multi-Agent Multi-Armed Bandit Problem 

This repository provides implementations for solving the multi-armed bandit problem using the Upper Confidence Bound (MUCB) and Learning with Linear Rewards (LLR) algorithms.

## Introduction

The Combinatorial Network Optimization multi-armed bandit problem is a learning theory challenge where rewards from multiple options, or "arms", are dependent and form a linear combination of unknown parameters, applicable to various network optimization tasks.

## Implemented Algorithm

### Learning with Linear Rewards (LLR) Algorithm

The `MUCB` class implements the Upper Confidence Bound algorithm. This algorithm is an extension of the standard UCB algorithm, designed to handle the multi-agent bandit problem. The core of the MUCB algorithm lies in its ability to balance exploration and exploitation, leveraging the Upper Confidence Bounds to guide its decisions.

The Learning with Linear Rewards (LLR) algorithm is a dynamic policy designed for multi-armed bandit problems where each arm yields stochastic rewards with unknown means. The LLR algorithm aims to minimize regret. Unlike many bandit algorithms, LLR is designed to handle dependent arms, where rewards are a linear combination of a set of unknown parameters. 
## Hungarian Algorithm 
The Hungarian method in the MUCB class implements a simplified version of the Hungarian algorithm. It finds the assignment of agents to arms that minimizes the total UCB, which is equivalent to maximizing the total expected reward. 

## Running the Scripts

### main.py

This script is the entry point for running the MUCB algorithm for a single simulation. It sets up the environment with a specified number of arms and their associated means and then runs the MUCB algorithm over a specified number of trials. The script concludes by plotting the average and cumulative regret over time.

You might want to adjust the following parameters according to your needs:

- `num_agents`: Number of agents in the simulation
- `num_arms`: Number of arms that can be chosen from
- `num_trials`: Number of trials for the simulation
- `mu`: List of means for each arm

To run the script, use:

```bash
python main.py
```
### smain.py

`smain.py` is used to run multiple simulations of the MUCB algorithm and compute average results over these simulations. This can be helpful in obtaining more stable and reliable performance estimates of the algorithm, as it accounts for the inherent randomness in the bandit problem. 

In `smain.py`, you can adjust the following parameters:

- `num_simulations`: The number of simulations to run.

To run the script, use the following command:

```bash
python smain.py
```
### Analysis

The graphs show the cumulative and average regret curve over time for a multi-agent bandit problem with 4 agents and 6 arms. The graph was created by running the experiment 20 times.

![Cumulative MUCB 20sims](https://github.com/DarkEyeX/bandit-algorithm/assets/43418679/aee8c6b4-0727-4569-b782-0855db0652b3)

The graph shows that the regret decreases over time as the agents learn more about the arms. As time steps increase the growth of total regret flattens after 40000 trials, which is noticeable, however there is still noise that is visible across.


![Average Regret MUCB 20sims](https://github.com/DarkEyeX/bandit-algorithm/assets/43418679/58efec06-cf80-49cb-8b93-d7bc462038ad)

The graph shows that the regret decreases over time as the agents learn more about the arms. The regret eventually plateaus because the agents are already exploiting the best arms as much as possible. However, the regret will never reach zero because there is always some uncertainty about which arm is the best.
