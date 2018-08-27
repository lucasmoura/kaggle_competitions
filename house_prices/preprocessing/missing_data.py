def fill_values_with_none(dataset, column_names):
    for column in column_names:
        dataset[column] = dataset[column].fillna('None')

    return dataset


def fill_values_with_zero(dataset, column_names):
    for column in column_names:
        dataset[column] = dataset[column].fillna(0)

    return dataset


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


def drop_values(dataset, column_names):
    for column in column_names:
        dataset = dataset.drop([column], axis=1)

    return dataset
