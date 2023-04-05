
#  UCB1 Algorithm for Multi-Armed Bandit

## Overview

This code implements the UCB1 (Upper Confidence Bound) algorithm for solving the Multi-Armed Bandit problem. The Multi-Armed Bandit problem is a classic reinforcement learning problem where an agent has to choose between multiple actions (or arms) to maximize its total reward over time.

The code consists of four main files:

- `main.py`: This is the main script that runs the UCB1 algorithm and generates plots for cumulative regret and average regret.
- `env.py`: This file defines the environment class, which simulates the environment of the Multi-Armed Bandit problem. It contains the true mean rewards of each arm and provides feedback to the agent in the form of rewards.
- `Gaussian_noise.py`: This file defines the Gaussian_noise class, which generates Gaussian noise for adding stochasticity to the rewards of the arms.
- `UCB1.py`: This file defines the UCB1 class, which implements the UCB1 algorithm for selecting arms and updating their estimated mean rewards.

## Code Flow

1. The `main.py` script initializes the number of arms (`num_arm`), the number of trials (`T`), and the true mean rewards of each arm (`mu`).
2. It creates an instance of the environment class (`env`) with the specified `num_arm` and `mu`.
3. It creates an instance of the UCB1 class (`UCB1`) with the specified `T` and `num_arm`.
4. It calls the `run()` method of the UCB1 class to run the UCB1 algorithm on the `env` environment.
5. During each trial, the UCB1 algorithm selects an arm based on its estimated mean reward and the exploration-exploitation trade-off determined by the UCB1 formula.
6. The `env` environment provides feedback in the form of rewards, and the UCB1 algorithm updates its estimated mean rewards and keeps track of cumulative regret and average regret.
7. After running the algorithm for `T` trials, the script generates plots for cumulative regret and average regret using the data collected during the algorithm's execution.
8. The plots are displayed using the matplotlib library.

## Usage

To run the code, you can execute the `main.py` script. You can modify the parameters such as `num_arm`, `T`, and `mu` to customize the experiment settings. The code will output plots for cumulative regret and average regret, which can be used to analyze the performance of the UCB1 algorithm in the Multi-Armed Bandit problem.

Note: Make sure to have the necessary dependencies such as numpy and matplotlib installed in your Python environment before running the code.
