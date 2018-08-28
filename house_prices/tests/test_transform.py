import unittest

import numpy as np
import pandas as pd

from preprocessing.transform import transform_categorical_column


class TestTransform(unittest.TestCase):

    def create_dummy_dataframe(self):
        dataset = pd.DataFrame({'a': ['a', 'b', 'c', 'd', 'e']})

        return dataset

    def test_transform_categorical_column(self):
        dataset = self.create_dummy_dataframe()

        column = 'a'
        map_dict = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
        transform_categorical_column(dataset, column, map_dict)
        expected_dataset = pd.DataFrame({'a': [1, 2, 3, 4, 5]})

        self.assertEqual(dataset[column].dtype, np.int64)
        self.assertTrue(dataset.equals(expected_dataset))
