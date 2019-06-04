from load_data import load_data
from collections import defaultdict
import argparse


def represents_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def find_baseline_attribute(data):
    best_ratio = 0
    best_attribute = None
    data_length = len(data)

    for attribute in data[0]:
        correct = 0
        if represents_integer(data[0][attribute]):
            pass
        else:
            counts = get_counts(data, attribute)
            for category in counts:
                if counts[category][0] > counts[category][1]:
                    correct += counts[category][0]
                else:
                    correct += counts[category][1]
       
        if correct / data_length > best_ratio:
            best_attribute = attribute
            best_ratio = correct / data_length

    counts = get_counts(data, best_attribute)
    labels = {}
    for category in counts:
        labels[category] = 0 if counts[category][0] > counts[category][1] else 1

    return best_attribute, labels


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
    sorted_data = sorted(data, key=lambda i: i[attribute])
    previous_value = 0
    max_information_gain = 0
    best_threshold = 0

    for x in range(len(data)):
        if data[x][attribute] != previous_value:
            threshold = data[x][attribute]
            find_information_gain(sorted_data, threshold, attribute)

        previous_value = data[x][attribute]


def find_information_gain(sorted_data, threshold, attribute):
    while sorted_data[attribute] < threshold:
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
print(find_baseline_attribute(data))
