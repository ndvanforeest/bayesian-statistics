import numpy as np

class Tree:
    min_size = 1
    max_depth = 0

    def __init__(self, X, Y, w=None, depth=0):
        self.X = X
        self.Y = Y
        if w is None:
            w = np.ones(len(Y))
        self.w = w / w.sum()
        self.depth = depth
        self.left, self.right = None, None
        self.split_col = None
        self.split_value = None
    

    def size(self):
        return len(self.Y)

    def p(self, z):
        return self.w[self.Y == z].sum()
    
    def entropy(self):
        res = np.array([self.p(z) for z in np.unique(self.Y)])
        return -res @ np.log(res)
    
    def gini(self):
        res = np.array([self.p(z) for z in np.unique(self.Y)])
        return (1 - res @ res) / 2
    
    def score(self):
        return self.gini()

    def proposed_split(self, col, value):
        s = self.X[:, col] < value
        left = Tree(self.X[s], self.Y[s], self.w[s], self.depth + 1)
        s = np.invert(s)
        right = Tree(self.X[s], self.Y[s], self.w[s], self.depth + 1)
        return left, right

    def set_optimal_split(self):
        n, p = self.X.shape
        best = np.infty
        best_row = 0
        best_col = 0
        for j in range(p):
            for i in range(n):
                left, right = self.proposed_split(col=j, value=self.X[i, j])
                score = left.score() + right.score()
                if score < best:
                    best, best_row, best_col = score, i, j
        self.split_col = best_col
        self.split_value = self.X[best_row, best_col]
        return self.proposed_split(self.split_col, self.split_value)

    def split(self):
        if self.size() <= self.min_size or self.depth >= self.max_depth:
            return
        left, right = self.set_optimal_split()
        if left.size() == 0 or right.size() == 0:
            return
        self.left, self.right = left, right
        self.left.split()
        self.right.split()
    

    def terminal(self):
        return self.left == None or self.right == None
    
    def majority_vote(self):
        values, counts = np.unique(self.Y, return_counts=True)
        return values[np.argmax(counts)]
    

    def predict(self, x):
        if self.terminal():
            return self.majority_vote()
        if x[self.split_col] < self.split_value:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

    def __repr__(self):
        d = " " * self.depth
        if not self.terminal():
            res = f"{d}X[{self.split_col}] < {self.split_value}\n"
            res += f"{d}L {self.left}\n"
            res += f"{d}R {self.right}\n"
        else:
            res = f"{d}T [{self.majority_vote()}]"
        return res
    

def test_gini(Y):
    values, counts = np.unique(Y, return_counts=True)
    return (1 - sum(counts ** 2) / len(Y) ** 2) / 2

def test():
    np.random.seed(3)
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

    Tree.max_depth = 3
    Tree.min_size = 1
    tree = Tree(X, Y, w=np.ones(len(Y)) / len(Y)) # set uniform weights
    print(tree.gini())
    print(test_gini(Y))
    tree.split()
    print(tree)

if __name__ == "__main__":
    test()
