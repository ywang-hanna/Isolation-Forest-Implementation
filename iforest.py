
# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd


class ExNode:
    def __init__(self, currDepth, size):
        self.currDepth = currDepth
        self.size = size
        # self.splitAttr = splitAttr
        # self.splitValue = splitValue


class InNode:
    def __init__(self, currDepth, splitAttr, splitValue, left, right):
        self.currDepth = currDepth
        self.splitAttr = splitAttr
        self.splitValue = splitValue
        self.left = left
        self.right = right


class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.n_nodes = 1

    def build_tree(self, X: np.ndarray, currDepth=0, improved=False):
        if currDepth >= self.height_limit or X.shape[0] <= 1:
            return ExNode(currDepth, X.shape[0])
        attrs = np.random.randint(0, X.shape[1], size=3)
        attr_list = []
        for attr in attrs:
            split = np.random.uniform(np.min(X[:, attr]),np.max(X[:, attr]))
            left_id = X[:, attr] < split
            left = X[left_id]
            right = X[np.invert(left_id)]
            attr_list.append((attr, min(len(left),len(right)), split, left, right))
        Attr = sorted(attr_list, key=lambda x: x[1])[0]
        splitAttr = Attr[0]
        minValue = np.min(X[:, splitAttr])
        maxValue = np.max(X[:, splitAttr])
        if minValue == maxValue:
            return ExNode(currDepth, X.shape[0])
        else:
            self.n_nodes += 2
            splitValue = Attr[2]
            X_left = Attr[3]
            X_right = Attr[4]
            return InNode(currDepth,
                          splitAttr,
                          splitValue,
                          left=self.build_tree(X_left, currDepth + 1, improved),
                          right=self.build_tree(X_right, currDepth + 1, improved))

    def fit(self, X: np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        self.root = self.build_tree(X, 0, improved)
        return self.root


class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees

    def fit(self, X: np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        trees = []
        height_limit = np.ceil(np.log2(self.sample_size))
        for i in range(self.n_trees):
            # sample_X = X[np.random.choice(X.shape[0], size=self.sample_size, replace=False), :]
            sample_X = X[np.random.randint(X.shape[0], size=self.sample_size), :]
            iTree = IsolationTree(height_limit)
            iTree.fit(sample_X, improved)
            trees.append(iTree)
        self.trees = trees
        return self

    def c(self, size):
        if size < 2: return 0
        if size == 2: return 1
        return 2 * (np.log(size - 1) + 0.5772156649) - 2 * (1 - 1 / size)

    def pathlength(self, x, node):
        if isinstance(node, ExNode):
            return node.currDepth + self.c(node.size)
        attr = node.splitAttr
        if x[attr] < node.splitValue:
            return self.pathlength(x, node.left)
        else:
            return self.pathlength(x, node.right)

    def path_length(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        lens = []
        if isinstance(X, pd.DataFrame):
            X = X.values
        for x in X:
            l = []
            for tree in self.trees:
                l.append(self.pathlength(x, tree.root))
            lens.append(np.mean(l))
        return np.asarray(lens)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        ci = self.c(self.sample_size)
        return 2 ** (-self.path_length(X) / ci)

    def predict_from_anomaly_scores(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return (scores >= threshold).astype(int)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        return self.predict_from_anomaly_scores(self.anomaly_score(X), threshold)


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    threshold = 1
    while True:
        y_hat = (scores >= threshold).astype(int)
        confusion = confusion_matrix(y, y_hat)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        if TPR >= desired_TPR: break
        threshold -= 0.01
    return threshold, FPR