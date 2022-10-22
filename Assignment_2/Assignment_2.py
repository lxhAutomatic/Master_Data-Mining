# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 16:16:40 2022

@author: Xinhao Lan
"""
from audioop import avg
from matplotlib.pyplot import text
import pandas as pd
import numpy as np
import os
import itertools
from time import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV


def read_data(path):
    """
    DESCRIPTION of the function

    :param path : TYPE. DESCRIPTION.

    return df_train : TYPE. DESCRIPTION.
    return df_test : TYPE. DESCRIPTION.
    """
    df = pd.DataFrame(columns=['text', 'fold', 'label'])
    counter = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            f = open(os.path.join(root, file), "r")
            if "truthful" in root:
                df.loc[counter] = [f.read(), root[-1], 0]
            else:
                df.loc[counter] = [f.read(), root[-1], 1]
            counter = counter + 1

    df_train = df.loc[df['fold'] != '5']
    df_train = df_train.reset_index(drop=True)
    df_test = df.loc[df['fold'] == '5']
    df_test = df_test.reset_index(drop=True)

    return df_train, df_test


def get_corpus(df):
    """
    creat a corpus from Dataframe

    :param df : Dataframe. Data from read_data()

    return corpus : list

    """
    corpus = []
    for i in range(0, len(df)):
        corpus.append(df.loc[i, 'text'])
    return corpus


def TF_IDF(df_train, df_test, ngram_range):
    """
    Extract features
    Term Frequency X Inverse Document Frequency.

    :param df : Dataframe. Data from read_data()

    return df : TYPE. DESCRIPTION.

    """
    corpus_train = get_corpus(df_train)
    corpus_test = get_corpus(df_test)
# Extracting features from the training data using a sparse vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram_range)
    X_train = vectorizer.fit_transform(corpus_train)
    X_train = pd.DataFrame(
        X_train.toarray(), columns=vectorizer.get_feature_names_out())

# Extracting features from the test data using the same vectorizer
    X_test = vectorizer.transform(corpus_test)
    X_test = pd.DataFrame(
        X_test.toarray(), columns=vectorizer.get_feature_names_out())

    return X_train, X_test


def data_preprocessing(path, ngram_range):
    df_train, df_test = read_data(path)

    y_train = np.array(df_train['label'])
    y_test = np.array(df_test['label'])

    X_train, X_test = TF_IDF(df_train, df_test, ngram_range=ngram_range)

    print("The number of extracted features: ", X_train.shape[1])
    print()

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    return X_train, y_train, X_test, y_test

# def MNB():
    # TODO fine-tune tehe hyper-parameter 'The number of those features'
    # TODO Implement the function for the Multinomial Naive Bayes.
    # TODO Test the different score with the default parameter and best parameter.

# def RLR():
    # TODO fine-tune the hyper-parameter λ (or C = 1/λ).
    # TODO Implement the function for the regularized logistic regression.
    # TODO Test the different score with the default parameter and best parameter.


def CT(x_train, y_train, x_test, y_test):
    t0 = time()
    print("Generating classification tree...")
    print()
    # calculate the alpha
    clf = DecisionTreeClassifier()
    # x_train = list(itertools.chain.from_iterable(x[:4]))
    # y_train = list(itertools.chain.from_iterable(y[:4]))
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
    # x_train = list(itertools.chain.from_iterable(x[:4]))
    # y_train = list(itertools.chain.from_iterable(y[:4]))
    # x_test = x[4]
    # y_test = y[4]
    clf = clf.fit(x_train, y_train)
    y_test_pre = clf.predict(x_test)
    print('With default ccp_alpha in classification tree, accuracy, precision, recall and f1 score on test sets:')
    print(accuracy_score(y_test, y_test_pre), precision_score(y_test, y_test_pre),
        recall_score(y_test, y_test_pre), f1_score(y_test, y_test_pre))

    # with best alpha
    clf = DecisionTreeClassifier(ccp_alpha=best_alpha)
    clf = clf.fit(x_train, y_train)
    y_test_pre = clf.predict(x_test)
    print('With best ccp_alpha in classification tree, accuracy, precision, recall and f1 score on test sets:')
    print(accuracy_score(y_test, y_test_pre), precision_score(y_test, y_test_pre),
        recall_score(y_test, y_test_pre), f1_score(y_test, y_test_pre))
    print()
    print("done in %0.3fs." % (time() - t0))
    print()


def RF(x, y, best_features, bigram=False):
    x_train = list(itertools.chain.from_iterable(x[:4]))
    y_train = list(itertools.chain.from_iterable(y[:4]))

    # TODO select features with the use of bigram.

    features_range = [best_features-2, best_features, best_features+2]
# features_range = [20, 40, 50, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 80, 100, 120]
    # 'n_estimators': [1000] fixed
    parameters = {'max_features': features_range}
    optimized_forest = RandomForestClassifier(n_estimators=1000)
    clf_test = GridSearchCV(optimized_forest, parameters, cv=4)
    clf_test.fit(x_train, y_train)
    rank_test = (clf_test.cv_results_['rank_test_score']).tolist()
    index = rank_test.index(min(rank_test))
    best_parameters = clf_test.cv_results_["params"][index]
    best_max_features = best_parameters["max_features"]
    for i, p in enumerate(clf_test.cv_results_["params"]):
        print("n features = ", p["max_features"], "\tavg_accuracy =", clf_test.cv_results_[
            "mean_test_score"][i], "\tstd_accuracy =", clf_test.cv_results_["std_test_score"][i])
    clf = RandomForestClassifier(
        n_estimators=1000, max_features=best_max_features)
    x_test = x[4]
    y_test = y[4]
    clf = clf.fit(x_train, y_train)
    y_test_pre = clf.predict(x_test)
    y_train_pre = clf.predict(x_train)
    print("Training Accuracy Random Forest (" + str(
        best_max_features) + " features): " + str(accuracy_score(y_train, y_train_pre)))
    print("Test Accuracy Random Forest (" + str(
        best_max_features) + " features): " + str(accuracy_score(y_test, y_test_pre)))


path = 'C:/Users/75581/Documents/GitHub/UU_Data_Mining_2022/Assignment_2/op_spam_v1.4/negative_polarity'
if ~os.path.exists(path):
    path = 'Assignment_2/op_spam_v1.4/negative_polarity'

print("Data(uni):")
X_train_uni, y_train, X_test_uni, y_test = data_preprocessing(path, ngram_range=(1, 1))

print("Data(uni+bi):")
X_train_uni_bi, y_train, X_test_uni_bi, y_test = data_preprocessing(path, ngram_range=(1, 2))

print("The number of training set: ", len(X_train_uni))
print("The number of test set: ", len(X_test_uni))
print()

print("CLF without bigram features added:")
CT(X_train_uni, y_train, X_test_uni, y_test)

print("CLF with bigram features added:")
CT(X_train_uni_bi, y_train, X_test_uni_bi, y_test)