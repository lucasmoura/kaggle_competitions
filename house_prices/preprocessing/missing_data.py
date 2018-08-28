def fill_nan_with_value(dataset, column_names, value):
    dict_map = dict.fromkeys(column_names, value)
    dataset.fillna(dict_map, inplace=True, downcast='infer')


def fill_values_with_mode(dataset, column_names):
    for column in column_names:
        dataset[column] = dataset[column].fillna(
                dataset[column].mode()[0])

    return dataset


def fill_values_with_median(dataset, column_names):
    for column in column_names:
        dataset[column] = dataset[column].fillna(
            dataset[column].median()[0])

    return dataset


def drop_columns(dataset, column_names):
    for column in column_names:
        dataset = dataset.drop([column], axis=1)

    return dataset
