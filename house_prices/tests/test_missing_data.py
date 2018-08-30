import unittest

import numpy as np
import pandas as pd

from preprocessing.missing_data import drop_columns, fill_nan_with_value


class MissingDataTest(unittest.TestCase):

    def create_dummy_dataframe(self, size):
        dataset = pd.DataFrame({'a': list(range(size)),
                                'b': list(range(size)),
                                'c': list(range(size)),
                                'd': list(range(size))})

        return dataset

    def test_drop_columns(self):
        dataset = self.create_dummy_dataframe(10)

        expected_columns = ['a', 'c']
        drop_columns(dataset, ['b', 'd'])
        self.assertEqual(dataset.columns.tolist(), expected_columns)

    def test_fill_nan_with_value(self):
        dataset = self.create_dummy_dataframe(10)

        dataset.iloc[[1, 2, 3], dataset.columns.get_loc('a')] = np.NaN
        fill_nan_with_value(dataset, ['a'], 0)

        expected_dataset = self.create_dummy_dataframe(10)
        expected_dataset.iloc[
            [1, 2, 3], expected_dataset.columns.get_loc('a')] = 0

        self.assertTrue(dataset.equals(expected_dataset))

        dataset = self.create_dummy_dataframe(10)

        dataset.iloc[[1, 2, 3], dataset.columns.get_loc('a')] = np.NaN
        fill_nan_with_value(dataset, ['a'], 'TS')

        expected_dataset = self.create_dummy_dataframe(10)
        expected_dataset.iloc[
            [1, 2, 3], expected_dataset.columns.get_loc('a')] = 'TS'

        self.assertTrue(dataset.equals(expected_dataset))
