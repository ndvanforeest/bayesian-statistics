import numpy as np
from numpy import linalg as LA
import cvxpy as cp
import matplotlib.pyplot as plt


class Predictor:
    def __init__(self, X, Y, gamma=0):
        self.X = X
        self.Y = Y
        self.gamma = gamma
        self.beta_0 = 0

    def loss(self, X, Y, beta):
        return np.square(Y - X @ beta).sum()

    def solve(self):
        raise NotImplemented

    def predict(self, x):
        return self.beta_0 + x @ self.beta

    def mse(self, X, Y):
        n, p = X.shape
        return self.loss(X, Y, self.beta).value / n


class OLS(Predictor):
    def solve(self):
        self.beta = LA.solve(self.X.T @ self.X, self.X.T @ self.Y)


class OLS_convex(Predictor):
    def loss(self, X, Y, beta):
        return cp.sum_squares(Y - X @ beta)

    def objective(self, beta):
        return self.loss(self.X, self.Y, beta)

    def solve(self):
        n, p = self.X.shape
        beta = cp.Variable(p)
        obj = cp.Minimize(self.objective(beta))
        problem = cp.Problem(obj)
        problem.solve()
        self.beta = np.array(beta.value)


def test(method):
    np.random.seed(3)

    n, p = 50, 20
    sigma = 5

    beta_star = np.ones(p)
    beta_star[:10] = np.zeros(10)

    X = np.random.randn(n, p)
    Y = X @ beta_star + np.random.normal(0, sigma, size=n)

    algo = method(X, Y, gamma=1)
    algo.solve()
    print(algo.beta[0])


def test_2(method):
    np.random.seed(3)

    n, p = 50, 20
    sigma = 5

    beta_star = np.ones(p)
    beta_star[:10] = np.zeros(10)

    X = np.random.randn(n, p)
    Y = X @ beta_star + np.random.normal(0, sigma, size=n)

    algo = method(X, Y, gamma=1)
    algo.solve()

    X = np.random.randn(2, p)
    print(algo.predict(X))


if __name__== "__main__":
    test(OLS)
    test(OLS_convex)
    test_2(OLS)
    test_2(OLS_convex)
