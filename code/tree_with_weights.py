import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs

from tree_simple import Tree as SimpleTree


def p(Y, w, z):
    res = w[Y == z].sum()
    if res == 0:
        return 0
    return res / w.sum()


def missclassify(Y, w):
    return 1 - max(p(Y, w, -1), p(Y, w, 1))


def gini(Y, w):
    return 1 - p(Y, w, -1) ** 2 - p(Y, w, 1) ** 2


class Tree(SimpleTree):
    def score(self, s):
        # return missclassify(self.Y[s], self.w[s])
        return gini(self.Y[s], self.w[s])

    def score_of_split(self, i, j):
        s = self.X[:, j] <= self.X[i, j]
        l_score = self.score(s) * self.w[s].sum() / self.w.sum()
        r_score = self.score(~s) * self.w[~s].sum() / self.w.sum()
        return l_score + r_score

    def fit(self, X, Y, sample_weight=None):
        if not self.test_input(X, Y):
            print("Tree: import of data failed")
            exit(1)
        self.X, self.Y = X, Y
        if sample_weight is None:
            sample_weight = np.ones(len(Y))
        self.w = sample_weight / sample_weight.sum()
        self.split()

    def majority_vote(self):
        if p(self.Y, self.w, -1) >= p(self.Y, self.w, 1):
            return -1
        return 1

    def test_input(self, X, Y):
        n, p = X.shape
        if len(Y) != n:
            print("Tree: len Y is not the same as the number of rows of X")
            return False
        if not set(np.unique(Y)).issubset([-1, 1]):
            print("Tree: The observations Y are not all equal to -1 or 1.")
            return False
        return True


def test_inputs():
    tree = Tree()
    X = np.arange(5).reshape(5, 1)
    Y = np.array([1, 0, 0, 1, 1])
    assert tree.test_input(X, Y) is False
    Y = 2 * Y - 1
    assert tree.test_input(X, Y) is True
    Y = np.array([1, 0, 0, 1])
    assert tree.test_input(X, Y) is False


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

    tree = Tree()
    tree.fit(X, Y)
    print(tree.predict(X))


def test_2():
    X = np.arange(5).reshape(5, 1)
    Y = np.array([1, 0, 0, 1, 1])
    Y = 2 * Y - 1
    n, p = X.shape
    w = np.ones(n)
    w[0] = 100
    w /= w.sum()

    tree = Tree()
    tree.fit(X, Y, sample_weight=w)
    my_y = tree.predict(X)

    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(X, Y, sample_weight=w)
    their_y = clf.predict(X)
    print((my_y == their_y).all())
    # print(clf.tree_.feature[0], clf.tree_.threshold[0], clf.tree_.impurity)
    # sk.tree.plot_tree(clf)
    # plt.show()


def test_3():
    np.random.seed(4)
    X, Y = make_blobs(n_samples=13, n_features=3, centers=2, cluster_std=20)
    Y = 2 * Y - 1
    n, p = X.shape
    w = np.random.uniform(size=n)
    w /= w.sum()

    tree = Tree()
    tree.fit(X, Y, sample_weight=w)
    my_y = tree.predict(X)

    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(X, Y, sample_weight=w)
    their_y = clf.predict(X)
    print((my_y == their_y).all())


if __name__ == "__main__":
    test_inputs()
    test()
    test_2()
    test_3()
