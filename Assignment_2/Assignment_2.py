# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 16:16:40 2022

@author: Xinhao Lan
"""
import pandas as pd
import numpy as np
import os
import itertools

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

def read_data(path):
    """
    DESCRIPTION of the function

    :param path : TYPE. DESCRIPTION.

    return df : TYPE. DESCRIPTION.

    """
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



# TODO use the algorithm to get the feature from those texts

#def MNB():
    # TODO fine-tune tehe hyper-parameter 'The number of those features'
    # TODO Implement the function for the Multinomial Naive Bayes.
    # TODO Test the different score with the default parameter and best parameter.
    
#def RLR():
    # TODO fine-tune the hyper-parameter λ (or C = 1/λ).
    # TODO Implement the function for the regularized logistic regression.
    # TODO Test the different score with the default parameter and best parameter.
    
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

    features_range = [best_features-2, best_features, best_features+2]
# features_range = [20, 40, 50, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 80, 100, 120]        
    parameters = {'max_features': features_range}                   # 'n_estimators': [1000] fixed
    optimized_forest = RandomForestClassifier(n_estimators=1000)
    clf_test = GridSearchCV(optimized_forest, parameters, cv=4)
    clf_test.fit(x_train, y_train)
    rank_test = (clf_test.cv_results_['rank_test_score']).tolist()
    index = rank_test.index(min(rank_test))
    best_parameters = clf_test.cv_results_["params"][index]
    best_max_features = best_parameters["max_features"]
    for i, p in enumerate(clf_test.cv_results_["params"]):
        print("n features = ", p["max_features"], "\tavg_accuracy =", clf_test.cv_results_["mean_test_score"][i], "\tstd_accuracy =", clf_test.cv_results_["std_test_score"][i] )
    clf = RandomForestClassifier(n_estimators=1000, max_features=best_max_features)
    x_test = x[4]
    y_test = y[4]
    clf = clf.fit(x_train, y_train)
    y_test_pre = clf.predict(x_test)
    y_train_pre = clf.predict(x_train)
    print("Training Accuracy Random Forest ("  + str(
        best_max_features) + " features): " + str(accuracy_score(y_train, y_train_pre)))
    print("Test Accuracy Random Forest (" + str(
        best_max_features) + " features): " + str(accuracy_score(y_test, y_test_pre)))
    
path = 'C:/Users/75581/Documents/GitHub/UU_Data_Mining_2022/Assignment_2/op_spam_v1.4/negative_polarity'
df = read_data(path)
print(df)
    
    
    
        