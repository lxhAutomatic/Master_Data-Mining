# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 16:16:40 2022

@author: Xinhao Lan, Yanming Li
"""
from audioop import avg
from matplotlib.pyplot import text
import pandas as pd
import numpy as np
import os
from time import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlxtend.evaluate import mcnemar, mcnemar_table
from scipy.stats import chisquare
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt

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

    # y_train = np.array(df_train['label'])
    # y_test = np.array(df_test['label'])

    y_train = df_train['label']
    y_test = df_test['label']

    X_train, X_test = TF_IDF(df_train, df_test, ngram_range=ngram_range)

    print("The number of extracted features: ", X_train.shape[1])
    print()

    # X_train = np.array(X_train)
    # X_test = np.array(X_test)

    return X_train, y_train, X_test, y_test


def TF_IDF_new(df_train, df_test, ngram_range):
    """
    Extract features
    Term Frequency X Inverse Document Frequency.

    :param df : Dataframe. Data from read_data()

    return df : TYPE. DESCRIPTION.

    """
    corpus_train = get_corpus(df_train)
    df_train_fold1 = df_train.loc[df_train['fold'] == '1']
    df_train_fold1 = df_train_fold1.reset_index(drop=True)
    df_train_fold2 = df_train.loc[df_train['fold'] == '2']
    df_train_fold2 = df_train_fold1.reset_index(drop=True)
    df_train_fold3 = df_train.loc[df_train['fold'] == '3']
    df_train_fold3 = df_train_fold1.reset_index(drop=True)
    df_train_fold4 = df_train.loc[df_train['fold'] == '4']
    df_train_fold4 = df_train_fold1.reset_index(drop=True)
    corpus_train_fold1 = get_corpus(df_train_fold1)
    corpus_train_fold2 = get_corpus(df_train_fold2)
    corpus_train_fold3 = get_corpus(df_train_fold3)
    corpus_train_fold4 = get_corpus(df_train_fold4)
    corpus_test = get_corpus(df_test)
# Extracting features from the training data using a sparse vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram_range)
    X_train = vectorizer.fit_transform(corpus_train)
    # X_train = pd.DataFrame(
    #     X_train.toarray(), columns=vectorizer.get_feature_names_out())

    X_train_fold1 = vectorizer.transform(corpus_train_fold1).toarray()
    X_train_fold2 = vectorizer.transform(corpus_train_fold2).toarray()
    X_train_fold3 = vectorizer.transform(corpus_train_fold3).toarray()
    X_train_fold4 = vectorizer.transform(corpus_train_fold4).toarray()
    X_train = [X_train_fold1,X_train_fold2,X_train_fold3,X_train_fold4]

# Extracting features from the test data using the same vectorizer
    X_test = vectorizer.transform(corpus_test).toarray()
    # X_test = pd.DataFrame(
    #     X_test.toarray(), columns=vectorizer.get_feature_names_out())

    return X_train, X_test

def data_preprocessing_new(path, ngram_range):
    # This function is only for RF
    df_train, df_test = read_data(path)

    # y_train = np.array(df_train['label'])
    # y_test = np.array(df_test['label'])

    # y_train = df_train['label']
    y_test = np.array(df_test['label'])

    y_train_fold1 = np.array(df_train.loc[df_train['fold'] == '1']['label'])
    y_train_fold2 = np.array(df_train.loc[df_train['fold'] == '2']['label'])
    y_train_fold3 = np.array(df_train.loc[df_train['fold'] == '3']['label'])
    y_train_fold4 = np.array(df_train.loc[df_train['fold'] == '4']['label'])
    y_train = [y_train_fold1,y_train_fold2,y_train_fold3,y_train_fold4]

    X_train, X_test = TF_IDF_new(df_train, df_test, ngram_range=ngram_range)

    # print("The number of extracted features: ", X_train.shape[1])
    print()

    # X_train = np.array(X_train)
    # X_test = np.array(X_test)

    return X_train, y_train, X_test, y_test

def important_features_4_MNB(classifier,n=5):
    class_labels = classifier.classes_
    feature_names = classifier.feature_names_in_

    topn_class1 = sorted(zip(classifier.feature_count_[0], feature_names),reverse=True)[:n]
    topn_class2 = sorted(zip(classifier.feature_count_[1], feature_names),reverse=True)[:n]

    print("Important words in negative reviews")

    for coef, feat in topn_class1:
        print(class_labels[0], coef, feat)

    print("-----------------------------------------")
    print("Important words in positive reviews")

    for coef, feat in topn_class2:
        print(class_labels[1], coef, feat)

def important_features_4_RLR(X_train,model,n=5):
    importances = pd.DataFrame(data={
        'Attribute': X_train.columns,
        'Importance': model.coef_[0]
    })
    importances = importances.sort_values(by='Importance', ascending=True)
    print("Important words in negative reviews")
    print(importances.head(n))
    importances = pd.DataFrame(data={
        'Attribute': X_train.columns,
        'Importance': model.coef_[0]
    })
    importances = importances.sort_values(by='Importance', ascending=False)
    importances = importances
    print("Important words in positive reviews")
    print(importances.head(n))

def important_features_4_RF_N_CT(X_train,model,n=5):
    explainer = shap.TreeExplainer(model, feature_perturbation='interventional',check_additivity=False)
    shap_values = explainer.shap_values(X_train)
    plt.figure(1)
    shap.summary_plot(shap_values=shap_values[0],
                features=X_train,
                feature_names=X_train.columns,
                plot_type='bar',
                max_display = n,
                title = "Important words in negative reviews",
                show=False
                )
    plt.title("Important words in negative reviews")
    plt.show()

    plt.figure(2)
    shap.summary_plot(shap_values=shap_values[1],
                features=X_train,
                feature_names=X_train.columns,
                plot_type='bar',
                max_display = n,
                title = "Important words in positive reviews",
                show=False
                )
    plt.title("Important words in positive reviews")
    plt.show()

    # plt.figure(3)
    # shap.summary_plot(shap_values=shap_values,
    #             features=X_train,
    #             feature_names=X_train.columns,
    #             # plot_type='bar',
    #             # max_display = n,
    #             title = "Important words in negative reviews",
    #             show=False
    #             )
    # plt.title("Important words in negative reviews")
    # plt.show()


def MNB(x_train, y_train, x_test, y_test):
    t0 = time()
    print("Generating Multinomial Naive Bayes...")
    print()
    nb = MultinomialNB()
    nb.fit(x_train, y_train)
    y_test_pre = nb.predict(x_test)
    print('With default alpha(1.0) in Multinomial Naive Bayes:')
    print(classification_report(y_test, y_test_pre))
    print()
    important_features_4_MNB(nb)

    params = {'alpha': [0.01, 0.1, 0.5, 0.7, 1.0, 10.0, ],
            }

    multinomial_nb_grid = GridSearchCV(MultinomialNB(), param_grid=params, n_jobs=-1, cv=5, verbose=5)
    multinomial_nb_grid.fit(x_train,y_train)

    print('Best Accuracy Through Grid Search : {:.3f}'.format(multinomial_nb_grid.best_score_))
    print('Best Parameters : {}\n'.format(multinomial_nb_grid.best_params_))

    # y_test_pre = multinomial_nb_grid.predict(x_test)

    nb_new = MultinomialNB(alpha = multinomial_nb_grid.best_params_['alpha'])
    nb_new.fit(x_train, y_train)
    y_test_pre_best = nb_new.predict(x_test)

    print('With best alpha of in Multinomial Naive Bayes:')
    print(classification_report(y_test, y_test_pre_best))
    print()
    important_features_4_MNB(nb_new)
    print()
    print("done in %0.3fs." % (time() - t0))
    print()

    return y_test_pre, y_test_pre_best

def RLR(x_train, y_train, x_test, y_test):
    t0 = time()
    clf = LogisticRegression().fit(x_train,y_train)
    y_test_pre = clf.predict(x_test)
    print('Without best 1/lambda in Regularized Logistic Regression:')
    # print(clf.get_params())
    print(classification_report(y_test, y_test_pre))
    print('1/lambda: {}\n'.format(clf.get_params()['C']))
    important_features_4_RLR(x_train,clf)

    clf = LogisticRegressionCV(cv=5, random_state=0).fit(x_train,y_train)
    y_test_pre_best = clf.predict(x_test)
    print('With best 1/lambda in Regularized Logistic Regression:')
    print(classification_report(y_test, y_test_pre_best))
    print('best 1/lambda: {}\n'.format(clf.get_params()['Cs']))
    important_features_4_RLR(x_train,clf)
    print()
    print("done in %0.3fs." % (time() - t0))
    print()
    return y_test_pre,y_test_pre_best

def CT(x_train, y_train, x_test, y_test):
    t0 = time()
    print("Generating classification tree...")
    print()
    # calculate the alpha
    clf = DecisionTreeClassifier(splitter= "random", random_state = 1, min_samples_leaf = 2)
    alphas = clf.cost_complexity_pruning_path(x_train, y_train)["ccp_alphas"]
    betas = list()
    for i, a in enumerate(alphas[:-1]):
        betas.append(np.math.sqrt(a * alphas[i + 1]))
    betas.append(np.inf)

    tree = DecisionTreeClassifier(splitter= "random", random_state = 1, min_samples_leaf = 2)
    parameters = {'ccp_alpha': betas}
    clf_test = GridSearchCV(tree, parameters, cv=4)
    clf_test.fit(x_train, y_train)
    rank_test = (clf_test.cv_results_['rank_test_score']).tolist()
    index_best_alpha = rank_test.index(min(rank_test))
    best_alpha = betas[index_best_alpha]
    print("Best alpha: ", best_alpha)

    # with default alpha 0.0
    clf = DecisionTreeClassifier(splitter= "random", random_state = 1, min_samples_leaf = 2)
    clf = clf.fit(x_train, y_train)
    y_test_pre = clf.predict(x_test)
    # print('With default ccp_alpha in classification tree, accuracy, precision, recall and f1 score on test sets:')
    # print(accuracy_score(y_test, y_test_pre), precision_score(y_test, y_test_pre),
    #     recall_score(y_test, y_test_pre), f1_score(y_test, y_test_pre))
    print('With default ccp_alpha in classification tree:')
    print(classification_report(y_test, y_test_pre))
    important_features_4_RF_N_CT(x_train,clf)
    # with best alpha
    clf_new = DecisionTreeClassifier(ccp_alpha=best_alpha, splitter= "random", random_state = 1, min_samples_leaf = 2)
    clf_new = clf_new.fit(x_train, y_train)
    y_test_pre_best = clf_new.predict(x_test)
    # print('With best ccp_alpha in classification tree, accuracy, precision, recall and f1 score on test sets:')
    # print(accuracy_score(y_test, y_test_pre), precision_score(y_test, y_test_pre),
    #     recall_score(y_test, y_test_pre), f1_score(y_test, y_test_pre))
    print('With best ccp_alpha in classification tree:')
    print(classification_report(y_test, y_test_pre_best))

    important_features_4_RF_N_CT(x_train,clf_new)
    print()
    print("done in %0.3fs." % (time() - t0))
    print()
    return y_test_pre,y_test_pre_best

def RF(x_train, y_train, x_test, y_test, best_features):
    features_range = [best_features-10, best_features-8, best_features-6, best_features-4, best_features-2, best_features, best_features+2, best_features+4, best_features+6, best_features+8, best_features+10]
    # 'n_estimators': [1000] fixed
    parameters = {'max_features': features_range}
    optimized_forest = RandomForestClassifier(n_estimators=1000, min_samples_leaf = 3)
    clf_test = GridSearchCV(optimized_forest, parameters, cv=4)
    clf_test.fit(x_train, y_train)
    rank_test = (clf_test.cv_results_['rank_test_score']).tolist()
    index = rank_test.index(min(rank_test))
    best_parameters = clf_test.cv_results_["params"][index]
    best_max_features = best_parameters["max_features"]
    for i, p in enumerate(clf_test.cv_results_["params"]):
        print("n features = ", p["max_features"], "\tavg_accuracy =", clf_test.cv_results_[
            "mean_test_score"][i], "\tstd_accuracy =", clf_test.cv_results_["std_test_score"][i])

    clf = RandomForestClassifier(n_estimators=1000, min_samples_leaf = 2)
    clf = clf.fit(x_train, y_train)
    y_test_pre = clf.predict(x_test)
    print('With best max_features in random forest:')
    print(classification_report(y_test, y_test_pre))
    important_features_4_RF_N_CT(x_train,clf)

    clf = RandomForestClassifier(n_estimators=1000, max_features=best_max_features, min_samples_leaf = 3)
    clf = clf.fit(x_train, y_train)
    y_test_pre_best= clf.predict(x_test)
    print()
    print('With default max_features in random forest:')
    print(classification_report(y_test, y_test_pre_best))
    important_features_4_RF_N_CT(x_train,clf)
    return y_test_pre,y_test_pre_best

def mcnemar_4_diff_models(y_test, y_pred_1, y_pred_2):
    # Significant test for different models
    tb_1 = mcnemar_table(y_test, y_pred_1, y_pred_2)
    chi, p = mcnemar(tb_1)
    print('chi-squared:', chi)
    print('p-value:', p)

path = 'C:/Users/75581/Documents/GitHub/UU_Data_Mining_2022/Assignment_2/op_spam_v1.4/negative_polarity'
if not os.path.exists(path):
    path = 'Assignment_2/op_spam_v1.4/negative_polarity'

print("Data(uni):")
X_train_uni, y_train, X_test_uni, y_test = data_preprocessing(path, ngram_range=(1, 1))

print("Data(uni+bi):")
X_train_uni_bi, y_train, X_test_uni_bi, y_test = data_preprocessing(path, ngram_range=(1, 2))
y_train = y_train.astype('int64')
y_test = y_test.astype('int64')
print("The number of training set: ", len(X_train_uni))
print("The number of test set: ", len(X_test_uni))
print()

# a_uni, b_uni, c_uni, d_uni = data_preprocessing_new(path, ngram_range=(1, 1))
# a_uni_bi, b_uni_bi, c_uni_bi, d_uni_bi = data_preprocessing_new(path, ngram_range=(1, 2))

# n_uni = int(np.sqrt(len(a_uni[0][0])))
# n_uni_bi = int(np.sqrt(len(a_uni_bi[0][0])))
# # print("RF without bigram features added:")
# # y_test_pre_uni, RF_y_test_pre_uni_best = RF(X_train_uni, y_train, X_test_uni, y_test, n_uni)
# print("RF with bigram features added:")
# y_test_pre_uni_bi, RF_y_test_pre_uni_bi_best = RF(X_train_uni_bi, y_train, X_test_uni_bi, y_test, n_uni_bi)
# mcnemar_4_diff_models(y_test,RF_y_test_pre_uni_best,RF_y_test_pre_uni_bi_best)


# print("CLF without bigram features added:")
# y_test_pre_uni, CLF_y_test_pre_uni_best = CT(X_train_uni, y_train, X_test_uni, y_test)
# print("CLF with bigram features added:")
# y_test_pre_uni_bi, CLF_y_test_pre_uni_bi_best = CT(X_train_uni_bi, y_train, X_test_uni_bi, y_test)
# mcnemar_4_diff_models(CLF_y_test,y_test_pre_uni_best,CLF_y_test_pre_uni_bi_best)

# print("MNB without bigram features added:")
# y_test_pre_uni, MNB_y_test_pre_uni_best = MNB(X_train_uni, y_train, X_test_uni, y_test)
# print("MNB with bigram features added:")
# y_test_pre_uni_bi, MNB_y_test_pre_uni_bi_best = MNB(X_train_uni_bi, y_train, X_test_uni_bi, y_test)
# mcnemar_4_diff_models(y_test,MNB_y_test_pre_uni_best,MNB_y_test_pre_uni_bi_best)

# print("RLR without bigram features added:")
# y_test_pre_uni, RLR_y_test_pre_uni_best = RLR(X_train_uni, y_train, X_test_uni, y_test)
# print("RLR with bigram features added:")
# y_test_pre_uni_bi, RLR_y_test_pre_uni_bi_best = RLR(X_train_uni_bi, y_train, X_test_uni_bi, y_test)
# mcnemar_4_diff_models(y_test,RLR_y_test_pre_uni_best,RLR_y_test_pre_uni_bi_best)


# print("mcnemar RLR MNB uni")
# mcnemar_4_diff_models(y_test,RLR_y_test_pre_uni_best,MNB_y_test_pre_uni_best)
# print("mcnemar RLR RF uni")
# mcnemar_4_diff_models(y_test,RLR_y_test_pre_uni_best,RF_y_test_pre_uni_best)
# print("mcnemar MNB RF uni")
# mcnemar_4_diff_models(y_test,MNB_y_test_pre_uni_best,RF_y_test_pre_uni_best)

# print("mcnemar RLR MNB uni_bi_")
# mcnemar_4_diff_models(y_test,RLR_y_test_pre_uni_bi_best,MNB_y_test_pre_uni_bi_best)
# print("mcnemar RLR RF uni_bi_")
# mcnemar_4_diff_models(y_test,RLR_y_test_pre_uni_bi_best,RF_y_test_pre_uni_bi_best)
# print("mcnemar MNB RF uni_bi_")
# mcnemar_4_diff_models(y_test,MNB_y_test_pre_uni_bi_best,RF_y_test_pre_uni_bi_best)