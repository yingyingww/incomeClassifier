from load_data import load_data
import decision_tree as dt
import argparse
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


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


def main():
    data = load_data('data/adult.data')
    tree = dt.build_decision_tree(data)
    baseline_tree = dt.build_decision_tree(data, max_depth=1)
    test_data = load_data('data/adult.test')

    baseline_metrics = compute_metrics(dt.decision_tree_classify, test_data, [baseline_tree])
    dt_metrics = compute_metrics(dt.decision_tree_classify, test_data, [tree])

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


if __name__ == "__main__":
    main()
