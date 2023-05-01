import numpy as np
from sklearn import gaussian_process
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from scipy.stats import norm
# For Updated Likelihood
from scipy.linalg import cholesky, cho_solve
from operator import itemgetter

from numpy.random import RandomState
import warnings

warnings.filterwarnings('ignore')


# ================== Synthetic Data Generation Functions ================== #

def x_data(dimension, bounds, training_budget):
    """ Uniformly sampled input data
    dimension - Data dimension; datatype - int
    bounds - bounds on space; data - [(lb,ub)]*dimension; lb,ub : datatype - float
    training_budget - Number of data points - datatype -int """
    lbs = np.array([bounds[i][0] for i in range(len(bounds))])
    ubs = np.array([bounds[i][1] for i in range(len(bounds))])
    x_dataset = (ubs - lbs) * np.random.random_sample(
        size=(training_budget, dimension)) + lbs  # uniformly sampled x_data
    return x_dataset


def partitions(tup, d, k, bounds):
    """ Generating uniformly sampled partitions for space
     tup - tuple of partitions along each dimension
     d - dimension
     k - number of classes
     bounds - boundaries of input space
     returns partitions - list of lists and label matrix - array"""
    # uniformly sampling each axis for point of partition
    par = list([])
    for i in range(d):
        # The size of the partition can be controlled by choosing the smallest size of the partition along each axis
        # which can be formed by using uniform dist over a array of number bound[0]:size:bound[1]
        x1 = np.random.uniform(bounds[i][0] + 0.01, bounds[i][1], size=(tup[i] - 1,))
        x2 = np.sort(
            np.append(x1, [bounds[i][0], bounds[i][1]]))  # Returns array with [lb, par, ub] - size size (tup[i]+1) x 1
        par += [x2]
    # Classes for each partition x selection of rows and y selection of columns for 2D
    label_mat = np.random.randint(k, size=tup)
    return par, label_mat


def checker_partitions(tup, d, k, bounds):
    """ Generating equisize partitions for space
     tup - tuple of partitions along each dimension
     d - dimension
     k - number of classes
     bounds - boundaries of input space
     returns partitions - list of lists and label matrix - array"""
    # Evenly sampling each axis for point of partition
    par = list([])
    for i in range(d):
        # The size of the partition can be controlled by choosing the smallest size of the partition along each axis
        # which can be formed by using uniform dist over a array of number bound[0]:size:bound[1]
        x1 = np.linspace(bounds[i][0], bounds[i][1],
                         tup[i] + 1)  # returns array with [lb, par, ub] - size (tup[i]+1) x 1
        par += [x1]

    label_mat = np.indices(tup).sum(axis=0) % k  # generating class label for each partition
    return par, label_mat


def checker_part_unbalanced(tup, d, k, bounds):
    """ Generating non uniform partitions for space
     tup - tuple of partitions along each dimension
     d - dimension
     k - number of classes
     bounds - boundaries of input space
     returns partitions - list of lists and label matrix - array"""
    # Unevenly sampling each axis for point of partition
    par = list([])
    for i in range(d):
        # The size of the partition can be controlled by choosing the smallest size of the partition along each axis
        # which can be formed by using uniform dist over a array of number bound[0]:size:bound[1]
        x1 = np.linspace(bounds[i][0], bounds[i][1],
                         2 ** tup[i])  # returns array with [lb, par, ub] - size (tup[i]+1) x 1
        par += [np.array([x1[j] for j in 2 ** np.arange(tup[i] + 1) - 1])]  # non uniform sized partitions

    label_mat = np.indices(tup).sum(axis=0) % k  # generating class label for each partition
    return par, label_mat


def label_fun(x, par, label_mat):
    """Label generation function for general partitions"""
    # x is the d dimensional input point - input shape (-1,)
    # par is the list of partition arrays along each dimension, len(par) == d, len(par[i]) == tup[i]+1
    # label_mat is the label matrix
    d = len(par)
    ind = ()
    for i in range(d):
        dim_ind = 1 * (x[i] <= par[i])
        ind_i = np.where(dim_ind == 1)[0][0] - 1
        if ind_i == -1:
            ind_i = 0
        ind += (ind_i,)
    c = label_mat[ind]
    return c


def gen_cls_dataset(x_dataset, par, label_mat):
    """Class dateset generation function for general partitions"""
    cls_dataset = np.empty((len(x_dataset, )))
    for i in range(len(x_dataset)):
        cls_dataset[i] = label_fun(x_dataset[i], par, label_mat)
    return cls_dataset.astype(int)


def gen_noisy_cls_dataset(x_dataset, par, label_mat, k):
    """Noisy Class dataset generation function for checkered partitions"""
    cls_dataset = np.empty((len(x_dataset, )))
    for i in range(len(x_dataset)):
        cls_dataset[i] = label_fun(x_dataset[i], par, label_mat)  # Noisy Case
    cls_dataset = cls_dataset + 0.5 * np.random.standard_normal(cls_dataset.shape)
    cls_dataset = np.rint(cls_dataset) % k
    return cls_dataset.astype(int)


def gen_y_dataset_har(x_dataset, cls_dataset, par_dataset, w, beta, c, alpha=0.1, std=0):
    """Generating the output using the harmonic functions"""
    y_dataset = np.cos(np.sum(x_dataset * w[cls_dataset], 1).reshape(-1, 1)) + c[par_dataset] \
                + std * (np.random.randn(cls_dataset.shape[0], 1))  # Intercept
    # y_dataset = np.exp(alpha * np.sum(x_dataset*beta[data_cls_ind], 1).reshape(-1, 1)) * \
    #     np.cos(np.sum(x_dataset * w[data_cls_nd], 1).reshape(-1, 1)) + c[data_par_ind]
    y_dataset = y_dataset.reshape(-1, )
    return y_dataset


def gen_y_levy(x_dataset, cls_dataset, par_dataset, w, beta, c, alpha=0.1, std=0):
    """Modified Levy Function for 2D"""
    y_dataset = np.empty((len(x_dataset, )))
    for i in range(len(x_dataset)):
        r = 1 + (x_dataset[i] - 1) / 4
        a = r[0] * w[cls_dataset[i], 0]
        b = r[1] * w[cls_dataset[i], 1]
        # y_dataset[i] = np.sin(np.pi * a) ** 2 + ((a - 1) ** 2) * (1 + 10 * (np.sin(np.pi * b) ** 2)) \
        #                + (b - 1) ** 2 + std * (np.random.randn())  # No Intercept
        y_dataset[i] = np.sin(np.pi * a) ** 2 + ((a - 1) ** 2) * (1 + 10 * (np.sin(np.pi * b) ** 2)) \
                       + (b - 1) ** 2 + c[par_dataset[i]] + std * (np.random.randn())  # Intercept
    y_dataset = y_dataset.reshape(-1, )
    return y_dataset


def gen_y_levy_cen(x_dataset, cls_dataset, par_dataset, par, partition_mat, w, beta, c, alpha=0.1, std=0):
    """Modified Partition Centered Levy Function for 2D"""
    y_dataset = np.empty((len(x_dataset, )))
    cen = [np.convolve(par[i], np.ones(2), 'valid') / 2 for i in range(len(par))]
    for i in range(len(x_dataset)):
        res = np.where(partition_mat == par_dataset[i])
        par_cen = np.array([cen[i][res[i][0]] for i in range(len(cen))])
        r = 1 + (x_dataset[i] - par_cen - 1) / 4
        a = r[0] * w[cls_dataset[i], 0]
        b = r[1] * w[cls_dataset[i], 1]
        # y_dataset[i] = np.sin(np.pi * a) ** 2 + ((a - 1) ** 2) * (1 + 10 * (np.sin(np.pi * b) ** 2)) + (b - 1) ** 2 \
        #                + std * (np.random.randn())  # no intercepts
        y_dataset[i] = np.sin(np.pi * a) ** 2 + ((a - 1) ** 2) * (1 + 10 * (np.sin(np.pi * b) ** 2)) + (b - 1) ** 2 + \
                       c[par_dataset[i]] + std * (np.random.randn())  # intercept
    y_dataset = y_dataset.reshape(-1, )
    return y_dataset


def gen_y_qing(x_dataset, cls_dataset, par_dataset, w, beta, c, alpha=0.1, std=0):
    """Modified Qing Function for 2D"""
    y_dataset = np.empty((len(x_dataset, )))
    for i in range(len(x_dataset)):
        a = x_dataset[i, 0] * w[cls_dataset[i], 0]
        b = x_dataset[i, 1] * w[cls_dataset[i], 1]
        # y_dataset[i] = ((a ** 2 - 1) ** 2 + ((b ** 2 - 2) ** 2)) + std * (np.random.randn()) # no intercepts
        y_dataset[i] = ((a ** 2 - 1) ** 2 + ((b ** 2 - 2) ** 2)) + c[par_dataset[i]] + std * (
            np.random.randn())  # noisy
    y_dataset = y_dataset.reshape(-1, )
    return y_dataset


def gen_y_qing_cen(x_dataset, cls_dataset, par_dataset, par, partition_mat, w, beta, c, alpha=0.1, std=0):
    """Modified Partition Centered Qing Function for 2D"""
    y_dataset = np.empty((len(x_dataset, )))
    cen = [np.convolve(par[i], np.ones(2), 'valid') / 2 for i in range(len(par))]
    for i in range(len(x_dataset)):
        res = np.where(partition_mat == par_dataset[i])
        par_cen = np.array([cen[i][res[i][0]] for i in range(len(cen))])
        a = (x_dataset[i, 0] - par_cen[0]) * w[cls_dataset[i], 0]
        b = (x_dataset[i, 1] - par_cen[1]) * w[cls_dataset[i], 1]
        # y_dataset[i] = ((a**2 - 1)**2 + ((b**2 - 2)**2)) + std * (np.random.randn()) no intercepts
        y_dataset[i] = ((a ** 2 - 1) ** 2 + ((b ** 2 - 2) ** 2)) + c[par_dataset[i]] + std * (
            np.random.randn())  # intercept
    y_dataset = y_dataset.reshape(-1, )
    return y_dataset


def gen_y_rosenbrock(x_dataset, cls_dataset, par_dataset, w, beta, c, alpha=0.1, std=0):
    """Modified Rosenbrock Function for 2D"""
    y_dataset = np.empty((len(x_dataset, )))
    for i in range(len(x_dataset)):
        a = x_dataset[i, 0] * w[cls_dataset[i], 0]
        b = x_dataset[i, 1] * w[cls_dataset[i], 1]
        # y_dataset[i] = 74 + 50 * ((b - a ** 2) **2) + (1 - a) ** 2 - 400 * np.exp(
        #   -(((a + 1) ** 2 + (b + 1) ** 2) / 0.1)) + std * (np.random.randn()) # no intercept
        y_dataset[i] = 74 + 50 * ((b - a ** 2) ** 2) + (1 - a) ** 2 - 400 * np.exp(
            -(((a + 1) ** 2 + (b + 1) ** 2) / 0.1)) + c[par_dataset[i]] + std * (np.random.randn())  # intercept
    y_dataset = y_dataset.reshape(-1, )
    return y_dataset


def gen_y_rosenbrock_cen(x_dataset, cls_dataset, par_dataset, par, partition_mat, w, beta, c, alpha=0.1, std=0):
    """Modified Partition Centered Rosenbrock Function for 2D"""
    y_dataset = np.empty((len(x_dataset, )))
    cen = [np.convolve(par[i], np.ones(2), 'valid') / 2 for i in range(len(par))]
    for i in range(len(x_dataset)):
        res = np.where(partition_mat == par_dataset[i])
        par_cen = np.array([cen[i][res[i][0]] for i in range(len(cen))])
        a = (x_dataset[i, 0] - par_cen[0]) * w[cls_dataset[i], 0]
        b = (x_dataset[i, 1] - par_cen[1]) * w[cls_dataset[i], 1]
        # y_dataset[i] = 74 + 50 * ((b - a ** 2) **2) + (1 - a) ** 2 - 400 * np.exp(
        #   -(((a + 1) ** 2 + (b + 1) ** 2) / 0.1)) + std * (np.random.randn()) # no intercept
        y_dataset[i] = 74 + 50 * ((b - a ** 2) ** 2) + (1 - a) ** 2 - 400 * np.exp(
            -(((a + 1) ** 2 + (b + 1) ** 2) / 0.1)) + c[par_dataset[i]] + std * (np.random.randn())  # intercept
    y_dataset = y_dataset.reshape(-1, )
    return y_dataset


def idx_thr(x, par):
    """ Function to generate the index and threshold for each datapoint"""
    c_par = []
    dist = []
    for i in range(len(x)):
        d1 = abs(par[i] - x[i])
        d2 = np.array(d1)
        c_par_i = np.argmin(d2)
        dist_i = np.min(d2)
        c_par += [c_par_i]
        dist += [dist_i]
    min_dist = min(dist)
    ind = dist.index(min_dist)
    thr = par[ind][c_par[ind]]
    return ind, thr


def gen_idx_thr(x_dataset, par):
    """ Function to compute index and threshold for the dataset"""
    idx_dataset = np.empty((len(x_dataset),))
    thr_dataset = np.empty((len(x_dataset),))
    for i in range(len(x_dataset)):
        idx, thr = idx_thr(x_dataset[i], par)
        idx_dataset[i] = idx
        thr_dataset[i] = thr
    return idx_dataset.astype(int), thr_dataset


# ========================== Tree Library ======================= #

class Node:
    """A decision tree node."""

    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class, bounds, depth=0, node_id=0):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.depth = depth
        self.node_id = node_id
        self.bounds = bounds
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.leaf = True
        self.leaf_id = None

    def preorder_traversal(self, node):
        """Preorder Traversal through the tree"""
        res = []
        if node:
            res.append(node)
            res = res + self.preorder_traversal(node.left)
            res = res + self.preorder_traversal(node.right)
        return res

    def traversing(self, node):
        """My version of Preorder traversal through the tree"""
        nodelist = []
        if not node.leaf:
            nodelist += [node]
            nodelist += self.traversing(node.left)
            nodelist += self.traversing(node.right)
        else:
            nodelist += [node]
        return nodelist

    def leaf_nodes(self, node):
        """Finding all the leaf nodes"""
        leafnodes = []
        if not node.leaf:
            leafnodes += self.leaf_nodes(node.left)
            leafnodes += self.leaf_nodes(node.right)
        else:
            leafnodes += [node]
        return leafnodes


# ======================== Decision Tree Classifier ======================= #

def leaf_node_data(X, root_node):
    """Partitioning the input data at leaf node"""
    node_rules = []
    node_rule = []

    def recurse(node, rule, rules):
        if not node.leaf:
            left_indices, right_indices = list(rule), list(rule)
            left_indices += [X[:, node.feature_index] <= node.threshold]  # True for left node indices - size - n x 1
            recurse(node.left, left_indices, rules)
            right_indices += [X[:, node.feature_index] > node.threshold]  # True for right node indices - size - n x 1
            recurse(node.right, right_indices, rules)
        else:
            rules += [rule]

    recurse(root_node, node_rule, node_rules)

    p = []
    for i in range(len(node_rules)):
        q = np.full(X[:, 0].shape, True)
        for j in range(len(node_rules[i])):
            q = np.logical_and(q, node_rules[i][j])
        p += [q]
    return p


class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.node_count = 0
        self.leaf_count = 0
        self.n_classes_ = 0
        self.n_features_ = 0
        self.tree_ = None

    def new_fit(self, X, y, indexes, thresholds, bounds, method='best'):
        """Fit the data 1st with the distance information and then based on gini index"""
        self.n_classes_ = len(set(y))  # classes are assumed to go from 0 to n-1
        self.n_features_ = X.shape[1]
        self.node_count = 0
        self.tree_ = self._tree(X, y, indexes, thresholds, bounds, method, select_list=None, depth=0)

    def predict(self, X):
        """Predict class for X."""
        return [self._predict(inputs) for inputs in X]

    def _gini(self, y):
        """Compute Gini impurity of a non-empty node"""
        m = y.size
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in range(self.n_classes_))

    def _best_split(self, X, y):
        """Find the best split for a node.
        "Best" means that the average impurity of the two children, weighted by their
        population, is the smallest possible. Additionally it must be less than the
        impurity of the current node.
        To find the best split, we loop through all the features, and consider all the
        midpoints between adjacent training samples as possible thresholds. We compute
        the Gini impurity of the split generated by that particular feature/threshold
        pair, and return the pair with smallest impurity.
        Returns:
            best_idx: Index of the feature for best split, or None if no split is found.
            best_thr: Threshold to use for the split, or None if no split is found.
        """
        # Need at least two elements to split a node.
        m = y.size
        if m <= 1:
            return None, None

        # Count of each class in the current node.
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]

        # Gini of current node.
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None

        # Loop through all features.
        for idx in range(self.n_features_):
            # Sort data along selected feature.
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):  # possible split positions
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes_))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_))

                # The Gini impurity of a split is the weighted average of the Gini impurity of the children.
                gini = (i * gini_left + (m - i) * gini_right) / m

                # The following condition is to make sure we don't try to split two
                # points with identical values for that feature, as it is impossible
                # (both have to end up on the same side of a split).
                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint

        return best_idx, best_thr

    def _best_split_new(self, X, y, list_idx_thr):
        """Find the best split for a node from the list of indexes and thresholds.
            best_idx: Index of the feature for best split, or None if no split is found.
            best_thr: Threshold to use for the split, or None if no split is found.
        """
        # Need at least two elements to split a node.
        m = y.size
        if m <= 1:
            return None, None

        # Count of each class in the current node.
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]

        # Gini of current node.
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None

        # Loop through all the indexes and thresholds
        for idx, threshold in list_idx_thr:
            # print(idx.astype(int), threshold)
            # Split the node according to each feature/threshold pair
            # and count the resulting population for each class in the children
            indices_left = X[:, idx.astype(int)] <= threshold
            y_left = y[indices_left]
            y_right = y[~indices_left]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            num_left = [np.sum(y_left == c) for c in range(self.n_classes_)]
            num_right = [np.sum(y_right == c) for c in range(self.n_classes_)]

            i = sum(num_left)
            gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes_))
            gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_))

            # The Gini impurity of a split is the weighted average of the Gini
            # impurity of the children.
            gini = (i * gini_left + (m - i) * gini_right) / m

            if gini < best_gini:
                best_gini = gini
                best_idx = idx
                best_thr = threshold  # midpoint

        return best_idx, best_thr

    def _tree(self, X, y, indexes, thresholds, bounds, method, select_list=None, depth=0):
        """ Build a initial decision tree from the given closest distance to boundary of data points
        X - Sampled data points
        y - labels of the datapoints
        indexes - Index along which the given point has the closest boundary
        thresholds - Point along the given index where the class label changes for each data point
        bounds - Boundaries of input space - [(lb, ub)]*d
        method - Method to select the split - best, median, ascending, descending, random
        select_list - list of ind,thr selected from the given data set for the split
        depth - depth of the current node
        """
        if select_list is None:
            select_list_ = []
        else:
            select_list_ = select_list.copy()
        list_idx_thr = sorted(set(zip(indexes, thresholds)))
        # Removing the feature indexes and thresholds that were previously selected for split
        for i in range(len(select_list_)):
            try:
                list_idx_thr.remove(select_list_[i])
            except ValueError:
                continue
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=self._gini(y),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
            bounds=bounds,
            depth=depth,
            node_id=self.node_count,
        )
        self.node_count += 1
        # Split recursively until max depth is reached reducing gini as much as possible
        if node.depth < self.max_depth and node.num_samples > self.min_samples_leaf:
            # if all the possible idx and thresholds have not been chosen
            if len(list_idx_thr) != 0:
                if method == 'best':
                    idx, thr = self._best_split_new(X, y, list_idx_thr)
                    select_list_.append((idx, thr))
                elif method == 'median':
                    j = len(list_idx_thr) // 2
                    idx, thr = self._best_split_new(X, y, [list_idx_thr[j]])
                    select_list_.append(list_idx_thr[j])
                elif method == 'ascending':
                    j = 0
                    idx, thr = self._best_split_new(X, y, [list_idx_thr[j]])
                    select_list_.append(list_idx_thr[j])
                elif method == 'descending':
                    j = -1
                    idx, thr = self._best_split_new(X, y, [list_idx_thr[j]])
                    select_list_.append(list_idx_thr[j])
                else:
                    j = np.random.randint(len(list_idx_thr))
                    idx, thr = self._best_split_new(X, y, [list_idx_thr[j]])
                    select_list_.append(list_idx_thr[j])
                # if the above split returns none then split based on usual best split
                if idx is None:
                    idx, thr = self._best_split(X, y)
            else:
                idx, thr = self._best_split(X, y)
            if idx is not None:
                # Bound[idx] -> [lb,ub] divided into bound_left[idx] -> [lb, thr) and bound_right[idx] -> [thr,ub]
                indices_left = X[:, idx] <= thr
                bounds_left = bounds.copy()
                bd_l = list(bounds_left[idx])  # Tuple objects are not changable
                bd_l[1] = thr
                bounds_left[idx] = tuple(bd_l)
                bounds_right = bounds.copy()
                bd_r = list(bounds_right[idx])  # Tuple objects are not changable
                bd_r[0] = thr
                bounds_right[idx] = tuple(bd_r)
                X_left, y_left, indexes_left, thresholds_left = \
                    X[indices_left], y[indices_left], indexes[indices_left], thresholds[indices_left]
                X_right, y_right, indexes_right, thresholds_right = \
                    X[~indices_left], y[~indices_left], indexes[~indices_left], thresholds[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.leaf = False
                node.left = self._tree(X_left, y_left, indexes_left, thresholds_left, bounds_left, method,
                                       select_list_, node.depth + 1)
                node.right = self._tree(X_right, y_right, indexes_right, thresholds_right, bounds_right, method,
                                        select_list_, node.depth + 1)
            else:
                node.leaf_id = self.leaf_count
                self.leaf_count += 1
        else:
            node.leaf_id = self.leaf_count
            self.leaf_count += 1
        return node

    def _predict(self, inputs):
        """Predict class for a single sample."""
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


# ========================= Modeling Functions ======================== #
"""Initializations"""
gradient_eval = True  # Evaluation of gradient of the function for minimization
GPR_CHOLESKY_LOWER = True  # Cholesky solve to compute the inverse in computation of posterior


def bo_par(x, y, c, idx, thr, kernel, bounds, max_depth, min_samp=2, restarts=1, normalize=True):
    """bo_par is a function to train a gps in each partition individually/independently"""

    """Training the Tree Classifier"""
    clf = DecisionTreeClassifier(min_samples_leaf=min_samp, max_depth=max_depth)
    clf.new_fit(x, c, idx, thr, bounds, method='best')

    """Finding the leaf nodes and corresponding classes"""
    clf_leaf_nodes = np.array(clf.tree_.leaf_nodes(clf.tree_))  # Array of leaf nodes
    clf_leaf_cls = np.array([i.predicted_class for i in clf_leaf_nodes])

    """Initializing GP's as many as leaves with leaf nodes as index"""
    gps = {k: gaussian_process.GaussianProcessRegressor(
        clone(kernel), normalize_y=normalize, n_restarts_optimizer=restarts) for k in clf_leaf_nodes}

    """Input dataPoints divided in to corresponding leaf datapoints"""
    clf_leaf_node_data = leaf_node_data(x, clf.tree_)

    """Training GP's in each region - stores data points belonging to the region and generates likelihood func"""
    for i in clf_leaf_nodes:
        if len(x[clf_leaf_node_data[i.leaf_id]]) >= 1:
            gps[i].fit(x[clf_leaf_node_data[i.leaf_id]], y[clf_leaf_node_data[i.leaf_id]])
        else:
            # print(len(x[clf_leaf_node_data[i.leaf_id]]))
            continue

    return clf, clf_leaf_nodes, clf_leaf_cls, clf_leaf_node_data, gps


def objective_func(theta, gps_cls, eval_gradient=gradient_eval):
    """Objective function - log marginal likelihood (lml) for each gp"""
    if eval_gradient:
        lml, grad = gps_cls.log_marginal_likelihood(theta, eval_gradient=eval_gradient, clone_kernel=False)
        return -lml, -grad
    else:
        return -gps_cls.log_marginal_likelihood(theta, clone_kernel=False)


def bo_hetero(x, y, c, idx, thr, kernel, bounds, theta_initial, max_depth, min_samp=2, restarts=1, normalize=True,
              weighted=True):
    """bo_hetero is a function for training gps in each partition with new lml objective
    that depends on the data of all the partitions that belong to the same class"""

    """Using bo_par to learn the classification tree and corresponding gps"""
    clf, clf_leaf_nodes, clf_leaf_cls, clf_leaf_node_data, gps = bo_par(x, y, c, idx, thr, kernel, bounds, max_depth,
                                                                        min_samp=min_samp, restarts=restarts,
                                                                        normalize=normalize)

    """New likelihood function == Sum of individual likelihoods of gp's belonging to the same class"""
    optima = []
    # print(list(set(clf_leaf_cls)))
    for ii in list(set(clf_leaf_cls)):
        optimum = []
        cls_leaf_nodes = clf_leaf_nodes[clf_leaf_cls == ii]
        gps_cls = [gps[i] for i in cls_leaf_nodes]
        gps_opt = [gps[i] for i in cls_leaf_nodes if hasattr(gps[i], "kernel_")]
        gps_bounds = gps_cls[0].kernel.bounds
        # print(ii, gps_cls, gps_bounds, theta_initial)

        if len(gps_opt) > 1:
            if not weighted:
                def objective_func1(theta, eval_gradient=gradient_eval):  # Summing up individual objective functions
                    if eval_gradient:
                        lml = sum([objective_func(theta, gp, eval_gradient=eval_gradient)[0] for gp in gps_opt])
                        grad = sum([objective_func(theta, gp, eval_gradient=eval_gradient)[1] for gp in gps_opt])
                        return lml, grad
                    else:
                        return sum([objective_func(theta, gp, eval_gradient=eval_gradient) for gp in gps_opt])
            else:
                def objective_func1(theta, eval_gradient=gradient_eval):  # Summing up individual objective functions
                    if eval_gradient:
                        lml = sum([gp.X_train_.shape[0] * objective_func(theta, gp, eval_gradient=eval_gradient)[0]
                                   for gp in gps_opt])
                        grad = sum([gp.X_train_.shape[0] * objective_func(theta, gp, eval_gradient=eval_gradient)[1]
                                    for gp in gps_opt])
                        return lml, grad
                    else:
                        return sum([gp.X_train_.shape[0] * objective_func(theta, gp, eval_gradient=eval_gradient)
                                    for gp in gps_opt])

            optimum += [(gps_opt[0]._constrained_optimization(objective_func1, theta_initial, bounds=gps_bounds))]

            # Additional runs are performed from log-uniform chosen initial theta
            if restarts > 0:  # Number of times the optimization is to be performed
                if not np.isfinite(gps_opt[0].kernel.bounds).all():
                    raise ValueError("Multiple optimizer restarts (n_restarts_optimizer>0) "
                                     "requires that all bounds are finite.")

                for iteration in range(restarts):
                    theta_initial1 = gps_opt[0]._rng.uniform(gps_bounds[:, 0], gps_bounds[:, 1])
                    optimum.append(gps_opt[0]._constrained_optimization(objective_func1, theta_initial1, gps_bounds))

            # Select result from run with minimal (negative) log-marginal likelihood
            lml_values = list(map(itemgetter(1), optimum))

            for i in gps_opt:
                i.kernel_.theta = optimum[np.argmin(lml_values)][0]
                """Updating the gps with the new kernel hyper-parameters"""
                K = i.kernel_(i.X_train_)
                K[np.diag_indices_from(K)] += i.alpha
                i.L_ = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
                # alpha_ = L^T \ (L \ y) which is used in prediction
                i.alpha_ = cho_solve((i.L_, GPR_CHOLESKY_LOWER), i.y_train_, check_finite=False)

            optima += [optimum[np.argmin(lml_values)]]
    return clf, clf_leaf_nodes, clf_leaf_cls, clf_leaf_node_data, gps


def bo_test(x, y, clf, clf_leaf_nodes, gps):
    """bo_test is the function to compute the mse in each partition"""
    p = leaf_node_data(x, clf.tree_)
    mse_par = []
    y_pred = []
    n_par = []
    for i in clf_leaf_nodes:
        if len(x[p[i.leaf_id]]) != 0:
            ypred = gps[i].predict(x[p[i.leaf_id]])
            err = mean_squared_error(ypred, y[p[i.leaf_id]])
            y_pred += [ypred]
            mse_par += [err]
            n_par += [len(x[p[i.leaf_id]])]  # number of test samples in each partition
        else:
            mse_par += [0]
            n_par += [len(x[p[i.leaf_id]])]
    return mse_par, n_par


# =================== Sampling Functions =================== #
def propose_location(acquisition, gps, clf_leaf_nodes, beta, d, n_restarts, initial_pt):
    """ Next sampling point based on the maximizer of the acquisition function
    acquisition - Sampling strategy
    gp - Gaussian Processes
    clf - Learnt Classification tree
    beta - constant to balance between exploration and exploitation
    bounds - [(lb,ub)]*d
    d - dimension
    n_restarts - no: of restarts for the optimizer
    initial_pt - starting point of optimizer if n_restarts = 0"""
    x_min_par = list([])
    ucb_min_par = list([])
    for m in clf_leaf_nodes:
        ucb_min_i = np.inf
        x0 = initial_pt.reshape(d, )
        x_min_i = initial_pt.reshape(d, )
        bounds = m.bounds
        # print(bounds)
        lbs = np.array([bounds[i][0] for i in range(len(bounds))]) + 1e-10
        ubs = np.array([bounds[i][1] for i in range(len(bounds))]) - 1e-10

        def obj_fun(x):
            # Objective function to be minimized (UCB maximization)
            return -acquisition(x.reshape(1, d), gps[m], beta)

        # Restart optimizer n_restart times
        if n_restarts == 0:
            result = minimize(obj_fun, x0=x0, bounds=bounds, method='L-BFGS-B')
            ucb_min_par += [result.fun]
            x_min_par += [result.x]
        else:
            for i in range(n_restarts + 1):
                x0 = (ubs - lbs) * np.random.random_sample(size=(d,)) + lbs
                # print(x0)
                result = minimize(obj_fun, x0, bounds=bounds, method='L-BFGS-B')
                if result.fun < ucb_min_i:
                    ucb_min_i = result.fun
                    x_min_i = result.x
            ucb_min_par += [ucb_min_i]
            x_min_par += [x_min_i]
    ucb_min = min(ucb_min_par)
    leaf_index = ucb_min_par.index(ucb_min)
    x_min = x_min_par[leaf_index]
    return x_min.reshape(1, d), ucb_min, leaf_index


def ucb_sampling(x, gp, beta):
    """Computing upper confidence bound at a given point."""
    mu, sig = gp.predict(x, return_std=True)
    ucb_x = mu + (np.sqrt(beta) * sig)
    return ucb_x


def ei_sampling(x, gps, y, epsilon=0):
    """EI sampling strategy
    x - Point of evaluation
    gp - Gaussian Process
    y - vector of all the function evaluations
    epsilon - to balance between exploration and exploitation"""
    mu, sig = gps.predict(x, return_std=True)
    y_max = max(y)
    z = ((mu - y_max - epsilon) / (sig + 1E-15))
    ei_x = (mu - y_max - epsilon) * norm.cdf(z) + sig * norm.pdf(z)
    return ei_x


def propose_location_gp(acquisition, gp, beta, d, n_restarts, initial_pt, bounds):
    """ Next sampling point based on the maximizer of the acquisition function
    acquisition - Sampling strategy
    gp - Gaussian Processes
    beta - constant to balance between exploration and exploitation
    bounds - [(lb,ub)]*d
    d - dimension
    n_restarts - no: of restarts for the optimizer
    initial_pt - starting point of optimizer if n_restarts = 0"""
    ucb_min = np.inf
    x0 = initial_pt.reshape(d, )
    lbs = np.array([bounds[i][0] for i in range(len(bounds))])
    ubs = np.array([bounds[i][1] for i in range(len(bounds))])

    def obj_fun(x):
        # Objective function to be minimized (UCB maximization)
        return -acquisition(x.reshape(1, d), gp, beta)

    # Restart optimizer n_restart times
    if n_restarts == 0:
        result = minimize(obj_fun, x0=x0, bounds=bounds, method='L-BFGS-B')
        x_min = result.x
    else:
        for i in range(n_restarts + 1):
            x0 = (ubs - lbs) * np.random.random_sample(size=(d,)) + lbs
            # print(x0)
            result = minimize(obj_fun, x0, bounds=bounds, method='L-BFGS-B')
            if result.fun < ucb_min:
                x_min = result.x
    return x_min.reshape(1, d), ucb_min
