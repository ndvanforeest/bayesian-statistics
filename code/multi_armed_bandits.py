import numpy as np
import matplotlib.pyplot as plt


p, q = 0.02, 0.05
eps = 0.03


class Strategy:
    def __init__(self, a_s=1, a_f=1, b_s=1, b_f=1, n=1000):
        self.a_s = a_s
        self.b_s = b_s
        self.a_f = a_f
        self.b_f = b_f
        self.n = n

        self.trace = np.zeros((n, 4))
        self.trace[0, :] = [a_s, a_f, b_s, b_f]

        np.random.seed(30)
        self.convert = np.zeros((n, 2))
        self.convert[:, 0] = np.random.binomial(1, p, size=n)
        self.convert[:, 1] = np.random.binomial(1, q, size=n)

    def update(self, winner, t):
        if winner == 0:
            self.a_s += self.convert[t, winner]
            self.a_f += 1 - self.convert[t, winner]
        else:
            self.b_s += self.convert[t, winner]
            self.b_f += 1 - self.convert[t, winner]
        self.trace[t, :] = [self.a_s, self.a_f, self.b_s, self.b_f]

    def run(self):
        raise NotImplemented

    def plot(self, fname):
        plt.clf()
        plt.ylim(0, 0.2)
        x_a = self.trace[:, 0] / (self.trace[:, 0] + self.trace[:, 1])
        x_b = self.trace[:, 2] / (self.trace[:, 2] + self.trace[:, 3])
        xx = np.arange(self.n) + 1
        average_reward = (self.trace[:, 0] + self.trace[:, 2]) / (xx + 1)  # prevent /0
        plt.plot(xx, x_a, label="a")
        plt.plot(xx, x_b, label="b")
        plt.plot(xx, p*np.ones(len(xx)), label="p")
        plt.plot(xx, q*np.ones(len(xx)), label="q")
        plt.plot(xx, average_reward, label="Reward")
        plt.legend()
        plt.savefig(fname)

    def run_and_plot(self, fname):
        self.run()
        self.plot(fname)


class Eps_greedy(Strategy):
    def run(self):
        flip = np.random.uniform(size=self.n)
        for t in range(1, self.n):
            p_hat = self.a_s / (self.a_s + self.a_f)
            q_hat = self.b_s / (self.b_s + self.b_f)
            winner = 0 if p_hat >= q_hat else 1
            winner = winner if flip[t] >= eps else 1 - winner
            self.update(winner, t)

class UCB1(Strategy):
    def run(self):
        for t in range(1, self.n):
            n_a = self.a_s + self.a_f
            n_b = self.b_s + self.b_f
            x_a = self.a_s / n_a + np.sqrt(2 * np.log(t) / n_a)
            x_b = self.b_s / n_b + np.sqrt(2 * np.log(t) / n_b)
            winner = 0 if x_a >= x_b else 1
            self.update(winner, t)

class Thompson(Strategy):
    def run(self):
        for t in range(1, self.n):
            x_a = np.random.beta(self.a_s, self.a_f)
            x_b = np.random.beta(self.b_s, self.b_f)
            winner = 0 if x_a >= x_b else 1
            self.update(winner, t)

if __name__ == "__main__":
    greedy = Eps_greedy(a_s=1, a_f=1, b_s=1, b_f=1, n=10000)
    greedy.run_and_plot("figures/policy_greedy.pdf")
    ucb = UCB1(a_s=1, a_f=1, b_s=1, b_f=1, n=10000)
    ucb.run_and_plot("figures/policy_ucb.pdf")
    thompson = Thompson(a_s=1, a_f=1, b_s=1, b_f=1, n=10000)
    thompson.run_and_plot("figures/policy_thompson.pdf")
