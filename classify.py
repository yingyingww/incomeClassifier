from load_data import load_data
from collections import defaultdict
import argparse


class Node:
    def __init__(self):
        self.left = None
        self.right = None


def decision_tree_classify(item, root):
    pass


def build_decision_tree(data):
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


def max_information_gain(data_length, counts):
    pass


def split_on_attribute(data, attribute, attribute_category):
    pass


def find_threshold(data, attribute):
    pass


def naive_bayes_classify():
    pass


def logistic_regression_classify():
    pass


def perceptron_classify():
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


data = load_data('data/adult.data')
print(get_counts(data, 'education'))
