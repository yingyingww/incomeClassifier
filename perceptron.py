from load_data import represents_integer, extract_features, get_labels
from collections import defaultdict

import math
import numpy as np

def perceptron(feature_vectors, labels, max_iter):
    weights = np.zeros(len(feature_vectors[0]))
    features_labels = [(feature_vectors[i], labels[i]) for i in range(len(feature_vectors))]
    for iterations in range(max_iter):
        np.random.shuffle(features_labels)
        for i in range(len(feature_vectors)):
            x = features_labels[i][0]
            true_label = features_labels[i][1]
            y=0
            if np.dot(x,weights) > 0:
                y=1
            if y != true_label:
                weights = weights + true_label * x

    return weights



def perceptron_test(test_features, weights):
    predicted_labels=[]
    for features in test_features:
        if np.dot(features,weights) > 0:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)

    return predicted_labels












