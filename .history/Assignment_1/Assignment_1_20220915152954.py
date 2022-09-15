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
        self.split = None
        self.left = None
        self.right = None
        self.value = value
        self.label = label
        self.space = ""

    def __str__(self):
        """
        A function is used to print the tree by using words.

        return None.
        """
        if self.split is None:
            if sum(self.label) > (len(self.label) - sum(self.label)):
                result = 1
            else:
                result = 0
            return("leaf node --> " + str(result))
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
        left.space = self.space + "\t"
        right.space = self.space + "\t"
        self.split = split
        self.left = left
        self.right = right

    def Gini_impurity(self):
        """
        This function is used to calculate the gini impurity

        return float. The result of gini impurity.
        """
        return (sum(self.label)/len(self.label)) * (1-(sum(self.label)/len(self.label)))

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
        flag = False
        features_list = np.sort(np.random.choice(
            np.arange(len(self.value[0])), size=nfeat, replace=False))
        for i in features_list:
            features = np.sort(np.unique(self.value[:, i]))
            for j, feature in enumerate(features[: -1]):
                thre = (features[j + 1] + feature)/2
                index_big = np.arange(len(self.value[:, i]))[
                    self.value[:, i] > thre]
                index_small = np.arange(len(self.value[:, i]))[
                    self.value[:, i] <= thre]
                gini_big = (sum(self.label[index_big])/len(index_big)) * \
                    (1-(sum(self.label[index_big])/len(index_big)))
                gini_small = (sum(self.label[index_small])/len(index_small)) * (
                    1-(sum(self.label[index_small])/len(index_small)))
                reduction = (self.Gini_impurity() - (gini_big*(len(index_big) /
                            len(self.label)) + gini_small*(len(index_small)/len(self.label))))
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
        #print(columns_list[i_temp] + " > " + str(thre_temp))
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
    root = Node(x, y)
    #temp = 0
    if len(y) < nmin:
        return root
    nodes = list()
    nodes.append(root)
    while nodes:
        node = nodes.pop()
        if node.Gini_impurity() > 0 and len(node.label) >= nmin:
            split, left, right = node.split_tree(minleaf, nfeat)
            if left != None:
                node.tree_update(split, left, right)
                nodes.append(left)
                nodes.append(right)
        #temp += 1
        # if temp == 2:
        #    print(root)
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
        index = np.random.choice(range(len(y)), size=len(y))
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
        y = []
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


def data_analysis(actual, predict):
    """
    This function is used to compute and print the accuracy, precision and recall of the result.

    :param actual: numpy.ndarray. The acutual labels of the test set.
    :param predict: numpy.ndarray. The predicted labels of the test set.
    """
    acc = accuracy_score(actual, predict)  # Calculate the accuracy score.
    print("Accuracy:", acc)
    p = precision_score(actual, predict)  # Calculate the precision score.
    print("Precision:", p)
    r = recall_score(actual, predict)  # Calculate the recall score.
    print("Recall:", r)
    cm = confusion_matrix(actual, predict)  # Calculate the confusion matrix.
    print("Confusion Matrix:")
    print(cm)


def data_preparation(path_1, path_2, metric_list, columns_name):
    """
    This function is used to prepare the data and do the selection on the data.

    :param path_1: str. Path for the eclipse data 2.0.
    :param path_2: str. Path for the eclipse data 3.0.
    :param metric_list: list. List contains the column index for the feature.
    :param columns_name: str. This parameter is used to judge if there are any post-release bugs that have been reported.

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
    path_1 = pathlib.Path(
        r"Assignment_1\Data\eclipse-metrics-packages-2.0.csv")
    path_2 = pathlib.Path(
        r"Assignment_1\Data\eclipse-metrics-packages-3.0.csv")
    metric_list = ["FOUT", "MLOC", "NBD", "PAR", "VG", "NOF",
                    "NOM", "NSF", "NSM", "ACD", "NOI", "NOT", "TLOC", "NOCU"]

    x_train, y_train, x_test, y_test = data_preparation(
        path_1, path_2, metric_list, "post")

    # Part 2.1
    clf = tree_grow(x_train, y_train, nmin=15, minleaf=5, nfeat=41)
    y_pred = tree_pred(x_test, clf)
    # print(clf)
    print("Part 2.1")
    data_analysis(y_test, y_pred)

    clf = tree_grow_b(x_train, y_train, nmin = 15, minleaf = 5, nfeat = 41, m= 100)
    y_pred = tree_pred_b(x_test, clf)
    print("Part 2.2")
    data_analysis(y_test, y_pred)

    clf = tree_grow_b(x_train, y_train, nmin = 15, minleaf = 5, nfeat = 6, m= 100)
    y_pred = tree_pred_b(x_test, clf)
    print("Part 2.3")
    data_analysis(y_test, y_pred)


"""
1. A short description of the data.
2. A picture of the first two splits of the single tree (the split in the root
node, and the split in its left or right child). Consider the classification
rule that you get by assigning to the majority class in the three leaf nodes
of this heavily simplified tree. Discuss whether this classification rule
makes sense, given the meaning of the attributes. (√)
3. Confusion matrices and the requested quality measures for all three models
(single tree, bagging, random forest).  (√)
4. A discussion of whether the differences in accuracy (that is, the proportion
of correct predictions) found on the test set are statistically significant.
Find a statistical test that is suited for this purpose.
"""
