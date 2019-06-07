from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler


def load_data(file_name, convert_strings=True):
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
    for item in data:
        item.pop('class', None)
    for item in test_data:
        item.pop('class', None)
    v = DictVectorizer()
    X_combined = v.fit_transform(data + test_data).toarray()
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X_combined)
    return scaled_X[0:len(data)], scaled_X[len(data):]
