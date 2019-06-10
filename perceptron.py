import numpy as np


def perceptron(feature_vectors, labels, max_iter):
    """Returns weights for the given feature vectors using the perceptron algorithm.

    Arguments:
    feature_vectors --- The feature matrix for the training data, where each row represents a 
    feature vector for a given data point.
    labels --- The labels (income class) of the training data. labels[i] corresponds to the 
    income class of the data point represented by feature_vectors[i].
    max_iter --- The maximum number of iterations allowed to be performed by the perceptron
    algorithm.
    """
    weights = np.zeros(len(feature_vectors[0]))
    features_labels = [(feature_vectors[i], labels[i]) for i in range(len(feature_vectors))]
    for iterations in range(max_iter):
        np.random.shuffle(features_labels)
        for i in range(len(feature_vectors)):
            x = features_labels[i][0]
            true_label = features_labels[i][1]
            y = 0
            if np.dot(x, weights) > 0:
                y = 1
            if y != true_label:
                weights = weights + true_label * x

    return weights


def perceptron_test(test_features, weights):
    """Classifies test_data (feature vectors) from weights calculated by the perceptron function.
    
    Arguments:
    test_features --- The feature matrix for the test data, where each row represents a feature
    vector for a data point in the test data.
    weights --- The weight vectors calculated by the perceptron function.
    """
    predicted_labels = []
    for features in test_features:
        if np.dot(features, weights) > 0:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)

    return predicted_labels



