import unittest

import pandas as pd

from kaggleflow.preprocessing.split import (split_data, create_folds)


class TestPreprocessinSplit(unittest.TestCase):

    def create_dummy_dataframe(self, size):
        dataset = pd.DataFrame({'a': list(range(size)),
                                'b': list(range(size))})

        return dataset

    def test_split_data(self):
        dataset = self.create_dummy_dataframe(10)

        expected_train_size = 8
        expected_test_size = 2
        train1, test1 = split_data(dataset)

        self.assertEqual(len(train1), expected_train_size)
        self.assertEqual(len(test1), expected_test_size)
        self.assertEqual(type(train1), type(dataset))

        train2, test2 = split_data(dataset)

        self.assertTrue(train1.equals(train2))
        self.assertTrue(test1.equals(test2))

        train3, test3 = split_data(dataset, seed=12)

        self.assertFalse(train1.equals(train3))
        self.assertFalse(test1.equals(test3))

    def test_create_folds(self):
        dataset = self.create_dummy_dataframe(10)
        num_folds = 5
        expected_folds = pd.Series({4: 2, 3: 2, 2: 2, 1: 2, 0: 2})

        fold_dataset = create_folds(dataset, num_folds)
        self.assertTrue(
            fold_dataset.fold.value_counts().sort_values().equals(
                expected_folds
            )
        )

        num_folds = 3
        expected_folds = pd.Series({2: 3, 1: 3, 0: 4})

        fold_dataset = create_folds(dataset, num_folds)
        self.assertTrue(
            fold_dataset.fold.value_counts().sort_values().equals(
                expected_folds
            )
        )

        dataset = self.create_dummy_dataframe(10)
        fold_dataset2 = create_folds(dataset, num_folds)
        self.assertTrue(fold_dataset.equals(fold_dataset2))

        dataset = self.create_dummy_dataframe(10)
        fold_dataset3 = create_folds(dataset, num_folds, seed=12)
        self.assertFalse(fold_dataset.equals(fold_dataset3))
