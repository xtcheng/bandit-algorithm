# Epsilon Greedy Algorithm

The Epsilon Greedy algorithm is a popular algorithm used in reinforcement learning to solve the exploration-exploitation dilemma. In this implementation, we use the Epsilon Greedy algorithm to solve a multi-armed bandit problem.

## Algorithm
The Epsilon-Greedy algorithm balances exploration and exploitation. With probability epsilon, the algorithm chooses a random arm (exploration), and with probability 1-epsilon, it chooses the arm with the highest expected reward (exploitation).

## Code Structure
The code is divided into three files: `env.py`, `epsilon.py`, and `epmain.py`. `env.py` defines the environment (arms and reward probabilities) and generates feedback for each arm. `epsilon.py` implements the Epsilon-Greedy algorithm and records the cumulative and average regret. `epmain.py` runs the simulation and plots the results.

## Usage
To run the code, simply run `epmain.py`. You can modify the parameters such as the number of arms, the number of trials, and the epsilon value to experiment with different scenarios.
