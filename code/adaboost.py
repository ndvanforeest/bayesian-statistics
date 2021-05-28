import numpy as np
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier


class AdaBoost:
    def __init__(self, rounds):
        self.rounds = rounds

    def fit(self, X, Y):
        n, p = X.shape
        w = np.ones(n) / n
        self.alphas = []
        self.learners = []

        for b in range(self.rounds):
            clf = DecisionTreeClassifier(max_depth=1)
            clf.fit(X, Y, sample_weight=w)
            self.learners.append(clf)
            y = clf.predict(X)

            error = w[y != Y].sum() / w.sum()
            alpha = 0.5 * np.log(1 / error - 1)
            self.alphas.append(alpha)
            w *= np.exp(-alpha * y * Y)

    def predict(self, x):
        res = sum(a * l.predict(x) for a, l in zip(self.alphas, self.learners))
        return np.sign(res)


def exp_loss(y, yhat):
    return np.exp(-y * yhat).sum() / len(y)


def test():
    np.random.seed(4)
    X, Y = make_blobs(n_samples=13, n_features=2, centers=2, cluster_std=20)
    Y = 2 * Y - 1  # don't forget this!

    ada = AdaBoost(8)
    ada.fit(X, Y)
    y = ada.predict(X)
    print(exp_loss(y, Y))


if __name__ == "__main__":
    test()
