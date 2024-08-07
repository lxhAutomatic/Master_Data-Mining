# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:20:34 2022

@author: Tingting Zhang, Xinhao Lan, Yanming Li.
"""
# Tingting Zhang (7278888)
# Xinhao Lan (1082620)
# Yanming Li (2033070)

# Import thrid party libraries
import numpy as np
import pandas as pd
import pathlib
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from mlxtend.evaluate import mcnemar, mcnemar_table
from scipy.stats import chisquare

columns_list = ["pre"]


class Node:
    """
    A class which represents each node in the decision tree.
    """

    def __init__(self, value, label):
        """
        A initialization function which initialize the entity.

        :param value: numpy.ndarray. Each row in the matrix contains the attribute values of one example.
        :param label: numpy.ndarray. Each row in the matrix contains the lable value of one example.

        return None.
        """
        self.split = None # a list contains two elements, the first is the feature name to split and the second is the thershold value. 
        self.left = None # a new node class which contains information of the left node.
        self.right = None # a new node class which contains inforamtion of the right node.
        self.value = value # saved feature value.
        self.label = label # saved label value.
        self.space = "" # this is used to better print the tree.

    def __str__(self):
        """
        A function is used to print the tree by using words.

        return None.
        """
        if self.split is None: # if this is a leaf node
            if sum(self.label) > (len(self.label) - sum(self.label)):
                result = 1
            else:
                result = 0
            return("leaf node --> " + str(result))
        # if this is not a leaf node, print the feature name, threshold, labels distribution.
        return("\n" + self.space + "Split feature name: " + columns_list[self.split[0]] + "; Split feature threshold " + str(self.split[1]) + "."
                + "\n" + self.space + "Total number examples: " +
                str(len(self.value))
                + ". " + str(sum(self.label)) + " labels are one and " +
                str(len(self.label) - sum(self.label)) + " labels are zero."
                + "\n" + self.space + "Left: " + str(self.left) + "\n" + self.space + "Right: " + str(self.right))

    def tree_update(self, split, left, right):
        """
        This function is used to update the split, left and right value for one entity

        :param split: list. The first element in the list is the feature index which is used to split.
                            The second element in the list is the threshold value whcih is used to wplit.
        :param left: Node. Another node class which contains the content of the left node of the current node.
        :param right: Node. Another node class which contains the content of the right node of the current node.

        return None.
        """
        left.space = self.space + "\t" # uodate the space before the left node.
        right.space = self.space + "\t" # update the space before the right node.
        self.split = split # update the split list.
        self.left = left # update the information of the left node.
        self.right = right # update the information of the right node.

    def Gini_impurity(self):
        """
        This function is used to calculate the gini impurity

        return float. The result of gini impurity.
        """
        return (sum(self.label)/len(self.label)) * (1-(sum(self.label)/len(self.label))) # compute the gini impurity.

    def split_tree(self, minleaf, nfeat):
        """
        This function is used to split the tree based on the gini impurity

        :param minleaf: int. The minimum number of observations required for a leaf node.
        :param nfeat: int. The number of features that should be considered for each split.

        return split: list. The first element in the list is the feature index which is used to split.
                            The second element in the list is the threshold value whcih is used to wplit.
        return left: Node. Another node class which contains the content of the left node of the current node.
        return right: Node. Another node class which contains the content of the right node of the current node.
        """
        i_temp = 0
        thre_temp = 0
        reduction_temp = 0
        # used to judge return none or not
        flag = False
        # select random feature set
        features_list = np.sort(np.random.choice(np.arange(len(self.value[0])), size=nfeat, replace=False))
        # loop the whole feature list
        for i in features_list:
            features = np.sort(np.unique(self.value[:, i]))
            # loop the feature to find the best split
            for j, feature in enumerate(features[: -1]):
                #calculate the threshold value
                thre = (features[j + 1] + feature)/2
                # find the index for the example which is bigger or smaller than the threshold
                index_big = np.arange(len(self.value[:, i]))[
                    self.value[:, i] > thre]
                index_small = np.arange(len(self.value[:, i]))[
                    self.value[:, i] <= thre]
                # caluculate the gini index
                gini_big = (sum(self.label[index_big])/len(index_big)) * \
                    (1-(sum(self.label[index_big])/len(index_big)))
                gini_small = (sum(self.label[index_small])/len(index_small)) * (
                    1-(sum(self.label[index_small])/len(index_small)))
                reduction = (self.Gini_impurity() - (gini_big*(len(index_big) /
                            len(self.label)) + gini_small*(len(index_small)/len(self.label))))
                # update the index, threshold and reduction, change the flag
                if reduction > reduction_temp and len(index_big) >= minleaf and len(index_small) >= minleaf:
                    i_temp = i
                    thre_temp = thre
                    reduction_temp = reduction
                    flag = True
        if flag == False:
            return None, None, None
        index_big = np.arange(len(self.value[:,  i_temp]))[
            self.value[:, i_temp] > thre_temp]
        index_small = np.arange(len(self.value[:,  i_temp]))[
            self.value[:, i_temp] <= thre_temp]
        split = [i_temp, thre_temp]
        left = Node(self.value[index_big], self.label[index_big])
        right = Node(self.value[index_small], self.label[index_small])

        return split, left, right


def tree_grow(x, y, nmin=None, minleaf=None, nfeat=None):
    """
    This function is used to grow a classification tree.

    :param x: numpy.ndarray. A data matrix (2-dimensional array) containing the attribute values.
                    Each row of x contains the attribute values of one training example.
    :param y: numpy.ndarray. The vector (1-dimensional array) of class labels.
                    The class label is binary, with values coded as 0 and 1.
    :param nmin: int. If a node contains fewer cases than nmin, it becomes a leaf node.
    :param minleaf: int. The minimum number of observations required for a leaf node.
    :param nfeat: int. The number of features that should be considered for each split.

    return root: Node. The constructed decision tree.
    """
    # initalize the root node
    root = Node(x, y)
    # Judge the requiremrent of nmin parameter
    if len(y) < nmin:
        return root
    nodes = list()
    nodes.append(root)
    # loop until the node is empty
    while nodes:
        node = nodes.pop() # pop the current node
        if node.Gini_impurity() > 0 and len(node.label) >= nmin:
            #grow the tree when one split is done
            split, left, right = node.split_tree(minleaf, nfeat)
            if left != None:
            # update and append the node if the left node is not a leaf node
                node.tree_update(split, left, right)
                nodes.append(left)
                nodes.append(right)
    return root


def tree_pred(x, tr):
    """
    This function is used to predicted class labels for the cases in x.

    :param x: numpy.ndarray. A data matrix (2-dimensional array) containing the attribute values of the cases
                    for which predictions are required,
    :param tr: Node. A tree object created with the function tree_grow.

    return y: numpy.ndarray: The vector (1-dimensional array) of predicted class labels for the cases in x.
    """
    y = []
    for i in x:
        tr_temp = tr
        # continue while the left node is not a null
        while tr_temp.left:
            # value is bigger than the threshold value
            if i[tr_temp.split[0]] > tr_temp.split[1]:
                tr_temp = tr_temp.left
            # value is smaller than the threshold value
            else:
                tr_temp = tr_temp.right
        # calculate the majority label of the trees
        if sum(tr_temp.label) > (len(tr_temp.label) - sum(tr_temp.label)):
            label = 1
        else:
            label = 0
        y.append(label)
    return y


def tree_grow_b(x, y, nmin=None, minleaf=None, nfeat=None, m=None):
    """
    This function is used to grow a list of m classification tree.

    :param x: numpy.ndarray. A data matrix (2-dimensional array) containing the attribute values.
                    Each row of x contains the attribute values of one training example.
    :param y: numpy.ndarray. The vector (1-dimensional array) of class labels.
                    The class label is binary, with values coded as 0 and 1.
    :param nmin: int. If a node contains fewer cases than nmin, it becomes a leaf node.
    :param minleaf: int. The minimum number of observations required for a leaf node.
    :param nfeat: int. The number of features that should be considered for each split.
    :param m: int. The number of bootstrap samples to be drawn.

    return tree_list: list. A list of m constructed decision tree.
    """
    tree_list = []
    for i in range(m):
        # bootstrap
        index = np.random.choice(range(len(y)), size=len(y))
        # grow the tree and add it into the list
        tree_list.append(tree_grow(x[index], y[index], nmin, minleaf, nfeat))
    return tree_list


def tree_pred_b(x, trs):
    """
    This function is used to predicted class labels for the cases in x with a list of m constructed decision tree.

    :param x: numpy.ndarray. A data matrix (2-dimensional array) containing the attribute values of the cases
                    for which predictions are required,
    :param trs: list. A list of m constructed decision tree.

    return ys: numpy.ndarray: The vector (1-dimensional array) of predicted class labels for the cases in x.
    """
    ys = []
    for i in x:
        # for the y, it is almost the same as the code in tree_pred
        y = []
        # loop in the tree list
        for tree in trs:
            tr_temp = tree
            while tr_temp.left:
                if i[tr_temp.split[0]] > tr_temp.split[1]:
                    tr_temp = tr_temp.left
                else:
                    tr_temp = tr_temp.right
            if sum(tr_temp.label) > (len(tr_temp.label) - sum(tr_temp.label)):
                label = 1
            else:
                label = 0
            y.append(label)
        ys.append(max(set(y), key=y.count))
    return ys

def data_preparation(path_1, path_2, metric_list, columns_name):
    """
    This function is used to prepare the data and do the selection on the data.

    :param path_1: str. Path for the eclipse data 2.0.
    :param path_2: str. Path for the eclipse data 3.0.
    :param metric_list: list. List contains the column index for the feature.
    :param columns_name: str. This parameter is used to get if there are any post-release bugs that have been reported.

    return train_data_x: numpy.ndarray. Numpy matrix for the feature value in the training data.
    return train_data_y: numpy.ndarray. Numpy matrix for the label value in the training data.
    return test_data_x: numpy.ndarray. Numpy matrix for the feature value in the test data.
    return test_data_y: numpy.ndarray. Numpy matrix for the label value in the test data.
    """
    # read data from the  csv file
    train_data = pd.read_csv(path_1, sep=";")
    test_data = pd.read_csv(path_2, sep=";")
    global columns_list
    # get pre-release bugs and
    # FOUT MLOC NBD PAR VG NOF NOM NSF NSM ACD NOI NOT TLOC(*3) + NOCU (*1) + pre-release bugs = 41 predictor variables
    for j in metric_list:
        for i in train_data.columns:
            if j in i:
                columns_list.append(i)

    train_data_x = train_data[columns_list]
    train_data.loc[train_data[columns_name] > 0, columns_name] = 1
    train_data_y = train_data[columns_name]
    # change DataFrame into np.array
    train_data_x = np.array(train_data_x)
    train_data_y = np.array(train_data_y)

    test_data_x = test_data[columns_list]
    test_data.loc[test_data[columns_name] > 0, columns_name] = 1
    test_data_y = test_data[columns_name]
    # change DataFrame into np.array
    test_data_x = np.array(test_data_x)
    test_data_y = np.array(test_data_y)
    return train_data_x, train_data_y, test_data_x, test_data_y


if __name__ == '__main__':
# Relative paths may be different in different environments
    path_1 = pathlib.Path(
        r"Data\eclipse-metrics-packages-2.0.csv")
    if ~path_1.is_file():
        path_1 = pathlib.Path(
            r"Assignment_1\Data\eclipse-metrics-packages-2.0.csv")
    path_2 = pathlib.Path(
        r"Data\eclipse-metrics-packages-3.0.csv")
    if ~path_2.is_file():
        path_2 = pathlib.Path(
            r"Assignment_1\Data\eclipse-metrics-packages-3.0.csv")

    metric_list = ["FOUT", "MLOC", "NBD", "PAR", "VG", "NOF",
                    "NOM", "NSF", "NSM", "ACD", "NOI", "NOT", "TLOC", "NOCU"]

    x_train, y_train, x_test, y_test = data_preparation(
        path_1, path_2, metric_list, "post")

    # Part 2.1
    # Data prediction and analysis for the single tree
    clf = tree_grow(x_train, y_train, nmin=15, minleaf=5, nfeat=41)
    y_pred_1 = np.array(tree_pred(x_test, clf))
    print(clf)
    # Data prediction and analysis for the Bagging
    clf = tree_grow_b(x_train, y_train, nmin = 15, minleaf = 5, nfeat = 41, m= 100)
    y_pred_2 = np.array(tree_pred_b(x_test, clf))
    # Data prediction and analysis for the Random Forest
    clf = tree_grow_b(x_train, y_train, nmin = 15, minleaf = 5, nfeat = 6, m= 100)
    y_pred_3 = np.array(tree_pred_b(x_test, clf))
