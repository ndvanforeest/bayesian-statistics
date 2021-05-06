import numpy as np


def p(Y, z):
    return (Y == z).sum() / len(Y)


def missclassify(Y):
    res = np.array([p(Y, z) for z in np.unique(Y)])
    return 1 - res.max(initial=0)


def entropy(Y):
    res = np.array([p(Y, z) for z in np.unique(Y)])
    return -res @ np.log(res)


def gini(Y):
    res = np.array([p(Y, z) for z in np.unique(Y)])
    return 1 - res @ res


class Tree:
    min_size = 1
    max_depth = 1

    def __init__(self, depth=0):
        self.X, self.Y = None, None
        self.depth = depth
        self.left, self.right = None, None
        self.split_col = None
        self.split_value = None

    def size(self):
        return len(self.Y)

    def score(self, Y):
        # return missclassify(Y)
        return gini(Y)

    def score_of_split(self, i, j):
        s = self.X[:, j] <= self.X[i, j]
        l_score = self.score(self.Y[s]) * len(self.Y[s]) / len(self.Y)
        r_score = self.score(self.Y[~s]) * len(self.Y[~s]) / len(self.Y)
        score = ((self.Y[s] != -1).sum() + (self.Y[~s] != 1).sum()) / len(self.Y)
        print("een:", l_score + r_score, score)
        return l_score + r_score

    def find_optimal_split(self):
        n, p = self.X.shape
        # best_score = np.inf
        best_score = self.score(self.Y)
        best_row = None
        best_col = None
        for j in range(p):
            for i in range(n):
                score = self.score_of_split(i, j)
                if score < best_score:
                    best_score, best_row, best_col = score, i, j
                    # print("best", i, j, score)
        self.split_row = best_row
        self.split_col = best_col
        self.split_value = self.X[best_row, best_col]

    def split(self):
        if self.size() <= self.min_size or self.depth >= self.max_depth:
            return
        self.find_optimal_split()
        if self.split_col == None:
            return
        s = self.X[:, self.split_col] <= self.split_value
        if s.all() or (~s).all():
            return
        self.left = Tree(depth=self.depth + 1)
        self.left.fit(self.X[s], self.Y[s])
        self.right = Tree(depth=self.depth + 1)
        self.right.fit(self.X[~s], self.Y[~s])

    def fit(self, X, Y):
        self.X, self.Y = X, Y
        self.split()

    def terminal(self):
        return self.left == None or self.right == None

    def majority_vote(self):
        values, counts = np.unique(self.Y, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, x):
        if self.terminal():
            return self.majority_vote()
        if x[self.split_col] <= self.split_value:
            return self.left.predict(x)
        else:
            return self.right.predict(x)


def test():
    X = np.array(
        [
            [2.771244718, 1.784783929],
            [1.728571309, 1.169761413],
            [3.678319846, 2.81281357],
            [3.961043357, 2.61995032],
            [2.999208922, 2.209014212],
            [7.497545867, 3.162953546],
            [9.00220326, 3.339047188],
            [7.444542326, 0.476683375],
            [10.12493903, 3.234550982],
            [6.642287351, 3.319983761],
        ]
    )
    Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    Y = 2 * Y - 1

    Tree.max_depth = 1
    Tree.min_size = 1
    tree = Tree()
    tree.fit(X, Y)

    for x in X:
        print(tree.predict(x))


if __name__ == "__main__":
    test()
