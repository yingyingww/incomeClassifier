from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_data(file_name, convert_strings=True):
    """Loads census data from the file.
    """
    output_data = []
    input_data = open(file_name, 'r')
    for line in input_data:
        line_data = line.split(',')
        line_data = [line_data[i].strip() for i in range(len(line_data))]
        if len(line_data) != 15:
            break
        output_data.append({'age': line_data[0], 'workclass': line_data[1],
                            'fnlwgt': line_data[2], 'education': line_data[3],
                            'education-num': line_data[4], 'marital-status': line_data[5],
                            'occupation': line_data[6], 'relationship': line_data[7],
                            'race': line_data[8], 'sex': line_data[9],
                            'capital-gain': line_data[10], 'capital-loss': line_data[11],
                            'hours-per-week': line_data[12], 'native-country': line_data[13],
                            'class': 0 if line_data[14] == '<=50K' or line_data[14] == '<=50K.' else 1})

    input_data.close()
    if convert_strings:
        convert_strings_to_integers(output_data)
    return output_data


def convert_strings_to_integers(data):
    """
    Converts strings representing integers (values corresponding to continuous attributes) to 
    integers.
    """
    for item in data:
        for attribute in item:
            if represents_integer(item[attribute]):
                item[attribute] = int(item[attribute])


def represents_integer(s):
    """Return true if the string s is an integer, false otherwise.
    
    This is used for determining if an attribute is categorical or continuous.
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


def extract_features(data, test_data):
    """Extracts feature vectors from training and test data. Feature vectors for training and 
    test data are created simultaneously to ensure that feature vectors are the same length 
    (there are features present in training but not test data and vice versa). Feature vector
    values are scaled according to their z-score.

    Arguments:
    data, test_data --- The training and test data.

    Returns:
    (X_train, X_test, feature_names) where X_train is the feature matrix for the training data, 
    X_test is the feature matrix for the test data, and feature_names is the list of feature
    names corresponding to the features in the feature matrices.
    
    """
    for item in data:
        item.pop('class', None)
    for item in test_data:
        item.pop('class', None)
    v = DictVectorizer()
    X_combined = v.fit_transform(data + test_data).toarray()
    X_train, X_test = X_combined[0:len(data)], X_combined[len(data):]
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.fit_transform(X_test)
    return scaled_X_train, scaled_X_test, v.feature_names_


def get_labels(data):
    return np.array([item['class'] for item in data])
