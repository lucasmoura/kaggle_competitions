def transform_categorical_column(dataset, column, map_dict):
    """
        This function is used to turn an ordinal categorical column with
        string labels into numeric values.

        :param dataset: A pandas DataFrame.
        :param column: The column to be transformed
        :param map_dict: The dict that maps the labels for its numeric value
    """
    dataset.loc[:, column] = dataset[column].map(map_dict)
