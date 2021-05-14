import numpy as np
import numpy.linalg as LA
 import pandas as pd

df = pd.read_csv("data_by_year_o.csv")

X = df[["tempo"]].values
Y = df["popularity"].values


xx = X  # keep for plotting
X = np.c_[np.ones(len(Y)), X]  # put 1 in front

n, p = X.shape

K = 100

# check on size of K
if K>n:
    print("K is too large, i.e., K should not exceed the number n of data points.")
    exit(1)

loss = []
for k in range(1, K + 1):
    # integer indices of test samples
    test_ind = ((n / K) * (k - 1) + np.arange(1, n / K + 1) - 1).astype('int')
    train_ind = np.setdiff1d(np.arange(n), test_ind)
    # print(test_ind)
    # quit()
    # print(train_ind)

    X_train, y_train = X[train_ind, :], Y[train_ind]
    X_test, y_test = X[test_ind, :], Y[test_ind]

    betahat = LA.solve(X_train.T @ X_train, X_train.T @ y_train)
    loss.append(LA.norm(y_test - X_test @ betahat) ** 2)

print(sum(loss) / n)
