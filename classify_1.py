from load_data import load_data, extract_features, get_labels

import decision_tree as dt
import perceptron

import argparse
import time
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

#Naive bayes
from sklearn.naive_bayes import GaussianNB
import warnings
# warnings.filterwarnings('ignore')

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()



def compute_metrics(classifier, test_data, params):
    """Computes accuracy, precision, recall, and f1-score for the given classifier.

    Arguments:
    classifier --- A function which classifies an item in the test_data. It's first parameter must
    be the test_data data point.

    test_data --- A list of data points in the test data as output by load_data in load_data.py.

    params --- Any additional parameters taken by the classifier function, in order. For example,
    this function will invoke the function call classifier(item, param_1, param_2 ... param_k)
    where item is a data point from the test data and param_0 ... param_k are the 0th through
    kth indices of params.

    Returns: A 4-tuple (accuracy, precision, recall, f1-score)

    """
    correct = len([item for item in test_data if classifier(item, *params) == item['class']])
    y_true = [item['class'] for item in test_data]
    y_pred = [classifier(item, *params) for item in test_data]

    return correct / len(test_data), precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)


def build_lr_model(X, y):
    return LogisticRegression(solver='sag').fit(X, y)

def build_perceptron_ski(X,y):
    return Perceptron(eta0=1, random_state=1).fit(X,y)

def build_naive_bayes(X,y):
    return GaussianNB().fit(X,y)


def main():
    data = load_data('data/adult.data')
    baseline_tree = dt.build_decision_tree(data, max_depth=1)
    print('Building decision tree...')
    dt_start = time.time()
    tree = dt.build_decision_tree(data)
    print('Decision tree built in ' + str(time.time() - dt_start) + ' s.')

    test_data = load_data('data/adult.val')
    baseline_metrics = compute_metrics(dt.decision_tree_classify, test_data, [baseline_tree])
    dt_metrics = compute_metrics(dt.decision_tree_classify, test_data, [tree])
    
    y_train = get_labels(data)
    y_test = get_labels(test_data)

    features = extract_features(data, test_data)
    X_train = features[0]
    X_test = features[1]

    print('Building logistic regression model...')
    lr_start = time.time()
    lr_model = build_lr_model(X_train, y_train)
    print('Logistic regression model built in ' + str(time.time() - lr_start) + ' s.')

    lr_pred = lr_model.predict(X_test)

    #perceptron
    weights = perceptron.perceptron(X_train, y_train, 6)
    perceptron_pred=perceptron.perceptron_test(X_test,weights)

    #skilearn model's perceptron
    perceptron_ski = build_perceptron_ski(X_train, y_train)
    y_percep_pred = perceptron_ski.predict(X_test)
    '''
    Result:
    Accuracy: 0.8032061912658928
    Precision: 0.5655369538587178
    Recall: 0.7202288091523661
    F1 Score: 0.6335773101555352
    '''

    # Gaussian Naive Bayes
    naive_bayes_model = build_naive_bayes(X_train, y_train)
    y_naive_bayes_pred = naive_bayes_model.predict(X_test)

    '''
    Result:
    Accuracy: 0.48473680977826916
    Precision: 0.3092619027626165
    Recall: 0.9576183047321893
    F1 Score: 0.4675341161536021
    '''


    print('Baseline:')
    print('Accuracy: ' + str(baseline_metrics[0]))
    print('Precision: ' + str(baseline_metrics[1]))
    print('Recall: ' + str(baseline_metrics[2]))
    print('F1 Score: ' + str(baseline_metrics[3]))
    
    print('\nDecision Tree:')
    print('Accuracy: ' + str(dt_metrics[0]))
    print('Precision: ' + str(dt_metrics[1]))
    print('Recall: ' + str(dt_metrics[2]))
    print('F1 Score: ' + str(dt_metrics[3]))

    print('\nLogistic Regression:')
    print('Accuracy: ' + str([y_test[i] == lr_pred[i] for i in range(len(y_test))].count(True) / len(test_data)))
    print('Precision: ' + str(precision_score(y_test, lr_pred)))
    print('Recall: ' + str(recall_score(y_test, lr_pred)))
    print('F1 Score: ' + str(f1_score(y_test, lr_pred)))

    print('\nPerceptron Regression:')
    print('Accuracy: ' + str([y_test[i] == perceptron_pred[i] for i in range(len(y_test))].count(True) / len(test_data)))
    print('Precision: ' + str(precision_score(y_test, perceptron_pred)))
    print('Recall: ' + str(recall_score(y_test, perceptron_pred)))
    print('F1 Score: ' + str(f1_score(y_test, perceptron_pred)))

    print('\nPerceptron Regression (ski):')
    print('Accuracy: ' + str([y_test[i] == y_percep_pred[i] for i in range(len(y_test))].count(True) / len(test_data)))
    print('Precision: ' + str(precision_score(y_test, y_percep_pred)))
    print('Recall: ' + str(recall_score(y_test, y_percep_pred)))
    print('F1 Score: ' + str(f1_score(y_test, y_percep_pred)))

    print('\nNaive Bayes (ski):')
    print('Accuracy: ' + str([y_test[i] == y_naive_bayes_pred[i] for i in range(len(y_test))].count(True) / len(test_data)))
    print('Precision: ' + str(precision_score(y_test, y_naive_bayes_pred)))
    print('Recall: ' + str(recall_score(y_test, y_naive_bayes_pred)))
    print('F1 Score: ' + str(f1_score(y_test, y_naive_bayes_pred)))

    print("\nCross Validation")
#     from sklearn.model_selection import KFold  # for K-fold cross validation
#     from sklearn.model_selection import cross_val_score  # score evaluation
#     from sklearn.model_selection import cross_val_predict  # prediction
#     kfold = KFold(n_splits=10, random_state=22)  # split the data into 10 equal parts
# #    accuracy = []
#     std = []
#     classifiers = ['Decision Tree', 'Perceptron', 'Log', 'Naive Bayes' ]
#     models = [tree,perceptron_ski, lr_model,naive_bayes_model]
#
#     for model in models:
#         cv_result = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
#         cv_result = cv_result
#         xyz.append(cv_result.mean())
#         std.append(cv_result.std())
#         accuracy.append(cv_result)
#     models_dataframe = pd.DataFrame({'CV Mean': xyz, 'Std': std}, index=classifiers)
#     models_dataframe






if __name__ == "__main__":
    main()
