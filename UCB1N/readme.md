# UCB1N Algorithm

This is an implementation of the Upper Confidence Bound (UCB1) algorithm for the multi-armed bandit problem with Gaussian noise, which is referred to as UCB1N.

## Description

The UCB1N algorithm is a variation of the UCB1 algorithm that takes into account the noise in the reward distribution. It works by maintaining an upper confidence bound for each arm that depends on the estimated mean reward and the number of times the arm has been played. The algorithm selects the arm with the highest upper confidence bound at each iteration.

## Usage

To use the UCB1N algorithm, you need to create an instance of the `UCB1N` class and call its `run` method with an instance of the `environment` class as an argument. The `environment` class simulates the multi-armed bandit problem with Gaussian noise and provides the reward and baseline reward for each arm when it is played.
