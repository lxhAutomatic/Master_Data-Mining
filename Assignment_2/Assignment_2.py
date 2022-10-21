# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 16:16:40 2022

@author: Xinhao Lan
"""
import pandas as pd
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from skelarn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def read_data(path):
    df = pd.DataFrame(columns = ['text', 'fold', 'label'])
    counter = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            f = open(os.path.join(root, file), "r")
            if "truthful" in root:
                df.loc[counter] = [f.read(), root[-1], 0]
            else:
                df.loc[counter] = [f.read(), root[-1], 1]
            counter = counter + 1
    return df

path = 'C:/Users/75581/Documents/GitHub/UU_Data_Mining_2022/Assignment_2/op_spam_v1.4/negative_polarity'
df = read_data(path)
print(df)

# TODO use the algorithm to get the feature from those texts

def MNB():

def RLR():
    
def CT(x, y):
    # calculate the alpha
    clf = DecisionTreeClassifier()
    x_train = list(itertools.chain.from_iterable(x[:4]))
    y_train = list(itertools.chain.from_iterable(y[:4]))
    alphas = clf.cost_complexity_pruning_path(x_train, y_train)["ccp_alphas"]
    betas = list()
    for i, a in enumerate(alphas[:-1]):
        betas.append(np.math.sqrt(a * alphas[i + 1]))
    betas.append(np.inf)


    tree = DecisionTreeClassifier()
    parameters = {'ccp_alpha': betas}
    clf_test = GridSearchCV(tree, parameters, cv=4)
    clf_test.fit(x_train, y_train)
    rank_test = (clf_test.cv_results_['rank_test_score']).tolist()
    index_best_alpha = rank_test.index(min(rank_test))
    best_alpha = betas[index_best_alpha]
    print("Best alpha: ", best_alpha)
    
    
    # with default alpha 0.0
    clf = DecisionTreeClassifier()
    x_train = list(itertools.chain.from_iterable(x[:4]))
    y_train = list(itertools.chain.from_iterable(y[:4]))
    x_test = x[4]
    y_test = y[4]
    clf = clf.fit(x_train, y_train)
    y_test_pre = clf.predict(x_test)
    print('With default ccp_alpha in classification tree, accuracy, precision, recall and f1 score on test sets:')
    print(accuracy_score(y_test, y_test_pre), precision_score(y_test, y_test_pre), recall_score(y_test, y_test_pre), f1_score(y_test, y_test_pre))
    
    # with best alpha
    clf = DecisionTreeClassifier(ccp_alpha = best_alpha)
    clf = clf.fit(x_train, y_train)
    y_test_pre = clf.predict(x_test)
    print('With default ccp_alpha in classification tree, accuracy, precision, recall and f1 score on test sets:')
    print(accuracy_score(y_test, y_test_pre), precision_score(y_test, y_test_pre), recall_score(y_test, y_test_pre), f1_score(y_test, y_test_pre))

def RF(x, y, best_features, bigram = False):
    x_train = list(itertools.chain.from_iterable(x[:4]))
    y_train = list(itertools.chain.from_iterable(y[:4]))
    # TODO select features with the use of bigram.
    
    
    
    
    
        