
def load_data(file_name):
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
                            'class': 0 if line_data[14] == '<=50K' else 1})

    input_data.close()
    return output_data
