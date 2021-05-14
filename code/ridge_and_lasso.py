import numpy as np
from numpy import linalg as LA
import cvxpy as cp
import matplotlib.pyplot as plt
from ols import Predictor, test

class Ridge(Predictor):
    def solve(self):
        n, p = self.X.shape
        A = self.X.T @ self.X + self.gamma * np.eye(p)
        b = self.X.T @ self.Y
        self.beta = LA.solve(A, b)


class Ridge_convex(Predictor):
    def loss(self, X, Y, beta):
        return cp.pnorm(Y - X @ beta, p=2) ** 2

    def regularizer(self, beta):
        return cp.pnorm(beta, p=2) ** 2

    def objective(self, beta):
        return self.loss(self.X, self.Y, beta) + self.gamma * self.regularizer(beta)

    def solve(self):
        n, p = self.X.shape
        beta = cp.Variable(p)
        obj = cp.Minimize(self.objective(beta))
        problem = cp.Problem(obj)
        problem.solve(solver="SCS")
        self.beta = np.array(beta.value)


class Ridge_with_constant(Predictor):
    def solve(self):
        n, p = self.X.shape
        A = self.X.T @ self.X
        dum = np.ones((1, n)) @ self.X
        A -= dum.T @ dum / n
        A += self.gamma * np.eye(p)
        b = (self.X.T - dum.T @ np.ones((1, n)) / n) @ self.Y
        self.beta = solve(A, b)
        self.beta_0 = np.ones((1, n)) @ (self.Y - dum @ beta)


class Lasso(Ridge_convex):
    def regularizer(self, beta):
        return cp.norm1(beta)


def statistics_plotting(method):
    n, p = 100, 20
    sigma = 5

    beta_star = np.ones(p)
    beta_star[10:] = 0

    np.random.seed(3)
    X = np.random.randn(n, p)
    Y = X @ beta_star + np.random.normal(0, sigma, size=n)

    X_train, Y_train = X[:50, :], Y[:50]
    X_test, Y_test = X[50:, :], Y[50:]

    gamma_values = np.logspace(-2, 3, 50)
    train_errors, test_errors, beta_values = [], [], []
    for g in gamma_values:
        algo = method(X_train, Y_train, gamma=g)
        algo.solve()
        train_errors.append(algo.mse(X_train, Y_train))
        test_errors.append(algo.mse(X_test, Y_test))
        beta_values.append(algo.beta)
    # print(train_errors)

    plt.clf()
    plt.plot(gamma_values, train_errors, label="Train error")
    plt.plot(gamma_values, test_errors, label="Test error")
    plt.xscale("log")
    plt.legend(loc="upper left")
    plt.xlabel(r"$\gamma$", fontsize=16)
    plt.title("Mean Squared Error (MSE)")
    fname = "figures/" + algo.__class__.__name__ + "_mse.pdf"
    plt.savefig(fname)

    plt.clf()
    num_coeffs = len(beta_values[0])
    for i in range(num_coeffs):
        plt.plot(gamma_values, [wi[i] for wi in beta_values])

    plt.xlabel(r"$\gamma$", fontsize=16)
    plt.xscale("log")
    plt.title("Regularization Path")
    fname = "figures/" + algo.__class__.__name__ + "_betas.pdf"
    plt.savefig(fname)


if __name__ == "__main__":
    test(Ridge)
    test(Ridge_convex)
    statistics_plotting(Ridge_convex)
    statistics_plotting(Ridge_convex)
