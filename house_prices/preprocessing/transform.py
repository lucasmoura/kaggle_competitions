import numpy as np
import pandas as pd


def transform_categorical_column(dataset, column, map_dict):
    """
        This function is used to turn an ordinal categorical column with
        string labels into numeric values.

        :param dataset: A pandas DataFrame.
        :param column: The column to be transformed
        :param map_dict: The dict that maps the labels for its numeric value
    """
    dataset.loc[:, column] = dataset[column].map(map_dict)


def transform_column_into_categorical_dtype(dataset, column):
    dataset[column] = dataset[column].astype('category')


def transform_categorical_to_one_hot(dataset, column):
    if type(column) != list:
        column = [column]

    return pd.get_dummies(dataset, columns=column)


def transform_to_log_scale(dataset, column):
    dataset.loc[:, column] = np.log(dataset[column])
