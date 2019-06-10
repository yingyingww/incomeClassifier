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
import matplotlib.pyplot as plt

import warnings
# warnings.filterwarnings('ignore')


def parse_args():
    """Function for handling command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--rep', action='store_true', help='Conduct reduced error pruning on the decision tree. It\'s recommended to supply a maximum depth parameter when conducting rpe pruning as this algorithm can take a long time to run otherwise.')
    parser.add_argument('--csp', action='store_true', help='Conduct chi square pruning on the decision tree.')
    parser.add_argument('--depth', type=int, help='The maximum depth of the decision tree.')
    parser.add_argument('--plot', action='store_true', help='Plot the accuracy, precision, recall, and f1-score of the classifiers run.')
    parser.add_argument('--lr_top', type=int, help='Get the n largest positively weighted features from the logistic regression model.')
    parser.add_argument('--lr_bot', type=int, help='Get the n largest negatively weighted features from the logistic regression model.')
    parser.add_argument('--baseline_attribute', default='education', help='Specify an attribute for the baseline (default is education)')
    parser.add_argument('--depth_plot', action='store_true', help='Plot maximum depth vs. f-score (for decision tree) and exit')
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


def get_lr_top_weights(model, num_features, feature_names):
    """Gets the features with the highest weights (positive and negative) from a logistic regression model.

    Arguments:
    model --- The logistic regression model.
    num_features --- The number of features to return.
    feature_names --- The names associated with the feature vectors originally passed to the
    logistic regression model.

    Returns:
    (best, worst) --- Where best is a list of the num_features features with the highest positive
    weights in the logistic regression model, and worst a list of the num_features features
    with the largest negative weights in the model.
    """
    weights = np.array(model.coef_[0])
    best_indices = np.argpartition(weights, -num_features)[-num_features:]
    worst_indices = np.argpartition(weights, -num_features)[0:num_features]
    return [feature_names[index] for index in best_indices], [feature_names[index] for index in worst_indices]


def plot_metrics(metrics_baseline, metrics_dt, metrics_perceptron, metrics_lr, metrics_dtre=None, metrics_dtcs=None):
    """Creates a bar chart to display the accuracy, precision, recall, and f1-score of the 
    classifiers.

    Arguments:
    (accuracy, precision, recall, f1-score) tuples corresponding to the baseline, decision-tree,
    perceptron, logistic regression, and optionally decision tree w/ reduced error pruning 
    classifiers.
    """
    n_groups = 4

    plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.15
    opacity = 0.8

    plt.bar(index, metrics_baseline, bar_width, alpha=opacity, color='b', label='Baseline')
    plt.bar(index + bar_width, metrics_dt, bar_width, alpha=opacity, color='g', label='Decision Tree')
    plt.bar(index + 2 * bar_width, metrics_perceptron, bar_width, alpha=opacity, color='r', label='Perceptron')
    plt.bar(index + 3 * bar_width, metrics_lr, bar_width, alpha=opacity, color='y', label='Logistic Regression')
    if metrics_dtre is not None:
        plt.bar(index + 4 * bar_width, metrics_dtre, bar_width, alpha=opacity, color='c', label='Decision Tree (Reduced Error Pruning)')
    elif metrics_dtcs is not None:
        plt.bar(index + 4 * bar_width, metrics_dtcs, bar_width, alpha=opacity, color='m', label='Decision Tree (Chi-Square Pruning)')

    plt.title('Metrics by classifier')
    plt.xticks(index + bar_width, ('Accuracy', 'Precision', 'Recall', 'F1-score'))
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()

    data = load_data('data/adult.data')
    test_data = load_data('data/adult.test2')
    val_data = load_data('data/adult.val')

    if args.depth_plot:
        print('Calculating f1-scores for different depths...')
        depths, scores = dt.tune_max_depth(data, val_data)
        plt.plot(depths, scores)
        plt.ylabel('F1-score')
        plt.xlabel('Maximum Depth')
        plt.show()
        quit()

    baseline_tree = dt.build_decision_tree(data, max_depth=1, forced_attribute=args.baseline_attribute)
    print('Building decision tree...')
    dt_start = time.time()
    if args.depth is not None:
        tree = dt.build_decision_tree(data, max_depth=args.depth)
    else:
        tree = dt.build_decision_tree(data)

    print('Decision tree built in ' + str(time.time() - dt_start) + ' s.')

    baseline_metrics = compute_metrics(dt.decision_tree_classify, test_data, [baseline_tree])
    dt_metrics = compute_metrics(dt.decision_tree_classify, test_data, [tree])

    if args.rep:
        print('Pruning decision tree (reduced error)...')
        dtre_start = time.time()
        dt.reduced_error_prune(tree, val_data)
        print('Decision tree pruned (reduced error) in ' + str(time.time() - dtre_start) + ' s.')
        dtre_metrics = compute_metrics(dt.decision_tree_classify, test_data, [tree])
    elif args.csp:
        print('Pruning decision tree (chi-square)...')
        dtcs_start = time.time()
        dt.chi_square_prune(tree, data)
        print('Decision tree pruned (chi-square) in ' + str(time.time() - dtcs_start) + ' s.')
        dtcs_metrics = compute_metrics(dt.decision_tree_classify, test_data, [tree])

    y_train = get_labels(data)
    y_test = get_labels(test_data)

    features = extract_features(data, test_data)
    X_train = features[0]
    X_test = features[1]
    feature_names = features[2]
    print('Building logistic regression model...')
    lr_start = time.time()
    lr_model = LogisticRegression(solver='sag').fit(X_train, y_train)

    print('Logistic regression model built in ' + str(time.time() - lr_start) + ' s.')

    if args.lr_top is not None:
        print('Top weighted features in logistic regression model: ' + str(get_lr_top_weights(lr_model, args.lr_top, feature_names)[0]))
    if args.lr_bot is not None:
        print('Top negatively weighted features in logistic regression model: ' + str(get_lr_top_weights(lr_model, args.lr_bot, feature_names)[1]))

    lr_pred = lr_model.predict(X_test)

    weights = perceptron.perceptron(X_train, y_train, 10)
    perceptron_pred = perceptron.perceptron_test(X_test, weights)

    perceptron_metrics = [y_test[i] == perceptron_pred[i] for i in range(len(y_test))].count(True) / len(test_data), precision_score(y_test, perceptron_pred), recall_score(y_test, perceptron_pred), f1_score(y_test, perceptron_pred)
    lr_metrics = [y_test[i] == lr_pred[i] for i in range(len(y_test))].count(True) / len(test_data), precision_score(y_test, lr_pred), recall_score(y_test, lr_pred), f1_score(y_test, lr_pred)

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

    if args.rep:
        print('\nDecision Tree (w/ reduced error pruning):')
        print('Accuracy: ' + str(dtre_metrics[0]))
        print('Precision: ' + str(dtre_metrics[1]))
        print('Recall: ' + str(dtre_metrics[2]))
        print('F1 Score: ' + str(dtre_metrics[3]))
    elif args.csp:
        print('\nDecision Tree (w/ chi-square pruning):')
        print('Accuracy: ' + str(dtcs_metrics[0]))
        print('Precision: ' + str(dtcs_metrics[1]))
        print('Recall: ' + str(dtcs_metrics[2]))
        print('F1 Score: ' + str(dtcs_metrics[3]))

    print('\nPerceptron:')
    print('Accuracy: ' + str(perceptron_metrics[0]))
    print('Precision: ' + str(perceptron_metrics[1]))
    print('Recall: ' + str(perceptron_metrics[2]))
    print('F1 Score: ' + str(perceptron_metrics[3]))

    print('\nLogistic Regression:')
    print('Accuracy: ' + str(lr_metrics[0]))
    print('Precision: ' + str(lr_metrics[1]))
    print('Recall: ' + str(lr_metrics[2]))
    print('F1 Score: ' + str(lr_metrics[3]))

    if args.plot:
        metrics_baseline = (baseline_metrics[0], baseline_metrics[1], baseline_metrics[2], baseline_metrics[3])
        metrics_dt = (dt_metrics[0], dt_metrics[1], dt_metrics[2], dt_metrics[3])
        metrics_perceptron = (perceptron_metrics[0], perceptron_metrics[1], perceptron_metrics[2], perceptron_metrics[3])
        metrics_lr = (lr_metrics[0], lr_metrics[1], lr_metrics[2], lr_metrics[3])
        metrics_dtre = None
        if args.rep:
            metrics_dtre = (dtre_metrics[0], dtre_metrics[1], dtre_metrics[2], dtre_metrics[3])
        elif args.csp:
            metrics_dtcs = (dtcs_metrics[0], dtcs_metrics[1], dtcs_metrics[2], dtcs_metrics[3])
        plot_metrics(metrics_baseline, metrics_dt, metrics_perceptron, metrics_lr, metrics_dtre)


if __name__ == "__main__":
    main()
