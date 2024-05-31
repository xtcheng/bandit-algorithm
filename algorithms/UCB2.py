import numpy as np

class UCB2:
    def __init__(self, T, num_arm, alpha):
        self.T = T
        self.num_arm = num_arm
        self.alpha = alpha
        self.num_play = np.zeros(self.num_arm)
        self.sum_mu = np.zeros(self.num_arm)
        self.mu = np.zeros(self.num_arm)
        self.epochs_played = np.zeros(self.num_arm)
        self.r = np.zeros(self.num_arm)
        self.ucb = np.zeros(self.num_arm)
        self.sum_rgt = 0
        self.avg_rgt = np.zeros(self.T)
        self.cum_rgt = np.zeros(self.T)

    def run(self, env):
        for i in range(0, self.T):
            if i < self.num_arm:
                arm = i
            else:
                # calculate UCB2 for each arm
                a_n_r = np.sqrt((1 + self.alpha) * np.log((i+1)/self.epochs_played) / (2 * self.epochs_played))
                self.ucb = self.mu + a_n_r
                arm = np.argmax(self.ucb)

            # play arm j exactly tau(r_j+1) - tau(r_j) times
            tau = np.exp(1/(self.alpha+1))
            plays = np.round(tau * (self.r[arm] + 1 - self.epochs_played[arm]) - tau * (self.r[arm] - self.epochs_played[arm]))
            self.epochs_played[arm] += plays

            for _ in range(int(plays)):
                rwd, br = env.feedback(arm)
                self.sum_mu[arm] += rwd
                self.num_play[arm] += 1
                self.mu[arm] = self.sum_mu[arm] / self.num_play[arm]
                self.sum_rgt += (br - rwd)
                self.avg_rgt[i] += self.sum_rgt / (i + 1)
                self.cum_rgt[i] += self.sum_rgt
            i += int(plays)
            # set r_j <-- r_j + 1
            self.r[arm] += 1

    def get_avg_rgt(self):
        return self.avg_rgt

    def get_cum_rgt(self):
        return self.cum_rgt

    def clear(self):
        self.num_play = np.zeros(self.num_arm)
        self.sum_mu = np.zeros(self.num_arm)
        self.mu = np.zeros(self.num_arm)
        self.epochs_played = np.zeros(self.num_arm)
        self.r = np.zeros(self.num_arm)
        self.ucb = np.zeros(self.num_arm)
        self.sum_rgt = 0
        self.avg_rgt = np.zeros(self.T)
        self.cum_rgt = np.zeros(self.T)
