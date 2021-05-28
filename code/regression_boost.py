import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

class Boost:
    def __init__(self, gamma, rounds):
        self.rounds = rounds
        self.gamma = gamma

    def fit(self, X, Y):
        self.g0 = Y.mean()
        residuals = Y - self.g0
        self.learners = []

        for b in range(self.rounds):
            clf = DecisionTreeRegressor(max_depth=1)
            clf.fit(X, residuals)
            residuals -= self.gamma * clf.predict(X)
            self.learners.append(clf)

    def predict(self, x):
        y = self.g0
        y += self.gamma*sum(l.predict(x) for l in self.learners)
        return y

def test():
    X, Y = make_regression(n_samples=5, n_features=1, n_informative=1, noise=10)

    boost = Boost(gamma=0.05, rounds=10)
    boost.fit(X, Y)
    y = boost.predict(np.array([[3]]))
    print(y)

def make_plots():
    np.random.seed(3)
    X, Y = make_regression(n_samples=30, n_features=1, n_informative=1, noise=10)
    plt.plot(X, Y, "*", label="train")

    n = 100
    x = np.linspace(-2.5, 2, n).reshape(n, 1)

    rounds = 100
    for gamma in [0.05, 1]:
        boost = Boost(gamma=gamma, rounds=rounds)
        boost.fit(X, Y)
        y = boost.predict(x)
        plt.plot(x, y, label=f"$\gamma = {gamma}$")

    plt.legend()
    plt.show()
    # plt.savefig("boost_test.pdf")

if __name__ == "__main__":
    test()
    make_plots()
