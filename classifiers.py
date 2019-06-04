from load_data import load_data
from collections import defaultdict


def decision_tree_classify(data):
    pass


def get_counts(data, attribute):
    counts = defaultdict(lambda: [0, 0])
    for item in data:
        if item['class'] == 0:
            counts[item[attribute]][0] += 1
        else:
            counts[item[attribute]][1] += 1

    counts = {item: tuple(counts[item]) for item in counts}

    return counts


def max_information_gain(data, attribute, counts):
    pass


def split_on_attribute(data, attribute):
    pass


def naive_bayes_classify():
    pass


def logistic_regression_classify():
    pass


def perceptron_classify():
    pass


data = load_data('data/adult.data')
print(get_counts(data, 'education'))
