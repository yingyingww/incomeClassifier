from load_data import load_data
from collections import defaultdict
import argparse
import math


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
        if attribute == 'fnlwgt':
            continue
        correct = 0
        if represents_integer(data[0][attribute]):  # check if attribute is continuous
            counts = get_counts(data, attribute, find_threshold(data, attribute))
        else:
            counts = get_counts(data, attribute)

        for category in counts:
            correct += counts[category][0] if counts[category][0] > counts[category][1] else counts[category][1]
        
        if correct / data_length > best_ratio:
            best_attribute = attribute
            best_ratio = correct / data_length

    if represents_integer(data[0][best_attribute]):
        counts = get_counts(data, best_attribute, find_threshold(data, attribute))
    else:
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


def get_counts(data, attribute, threshold=None):
    if threshold is not None:
        return get_counts_threshold(data, attribute, threshold)
    counts = defaultdict(lambda: [0, 0])
    for item in data:
        if item['class'] == 0:
            counts[item[attribute]][0] += 1
        else:
            counts[item[attribute]][1] += 1

    counts = {item: tuple(counts[item]) for item in counts}

    return counts


def get_counts_threshold(data, attribute, threshold):
    counts = defaultdict(lambda: [0, 0])
    for item in data:
        if item['class'] == 0 and item[attribute] < threshold:
            counts['below'][0] += 1
        elif item['class'] == 0 and item[attribute] >= threshold:
            counts['above'][0] += 1
        elif item['class'] == 1 and item[attribute] < threshold:
            counts['below'][1] += 1
        else:
            counts['above'][1] += 1
    return counts

def max_information_gain(data_length, counts):
    pass


def split_on_attribute(data, attribute, attribute_category):
    pass


def find_threshold(data, attribute):
    sorted_data = sorted(data, key=lambda i: i[attribute])
    previous_class = -1
    max_info_gain = 0
    best_threshold = 0

    tested = []

    for x in range(len(data)):
        if sorted_data[x][attribute] not in tested and sorted_data[x]['class'] != previous_class:
            threshold = sorted_data[x][attribute]
            tested.append(threshold)
            threshold_info_gain = get_information_gain(sorted_data, attribute, threshold)
            if threshold_info_gain > max_info_gain:
                max_info_gain = threshold_info_gain
                best_threshold = threshold

        previous_class = sorted_data[x]['class']

    return best_threshold


def get_information_gain(sorted_data, attribute, threshold=None):
    counts_bt = [0, 0]  # bt = below threshold, at = above threshold
    counts_at = [0, 0]
    current_index = 0
    while sorted_data[current_index][attribute] < threshold:
        if sorted_data[current_index]['class'] == 0:
            counts_bt[0] += 1
        else:
            counts_bt[1] += 1
        current_index += 1
    
    for x in range(current_index, len(sorted_data)):
        if sorted_data[x]['class'] == 0:
            counts_at[0] += 1
        else:
            counts_at[1] += 1

    total_bt = (counts_bt[0] + counts_bt[1])
    total_at = (counts_at[0] + counts_at[1])

    
    if 0 in counts_bt or 0 in counts_at:
        return 0


    probability_bt = total_bt / len(sorted_data)
    probability_at = total_at / len(sorted_data)
    conditional_bt = (counts_bt[0] / total_bt) * math.log((counts_bt[0] / total_bt), 2) + (counts_bt[1] / total_bt) * math.log((counts_bt[1] / total_bt), 2)
    conditional_at = (counts_at[0] / total_at) * math.log((counts_at[0] / total_at), 2) + (counts_at[1] / total_at) * math.log((counts_at[1] / total_at), 2)

    return probability_bt * -conditional_bt + probability_at * -conditional_at
            

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
print(find_threshold(data, 'age'))
print(find_baseline_attribute(data))
