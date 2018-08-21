import argparse

import numpy as np
import pandas as pd

from preprocessing.dataset import Dataset
from model.linear_regression.estimator import LinearRegressionEstimator


def create_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t',
                        '--train-file',
                        type=str,
                        help='Location of the train file (csv)')

    parser.add_argument('-ts',
                        '--test-file',
                        type=str,
                        help='Location of the test file (csv)')

    parser.add_argument('-np',
                        '--num-epochs',
                        type=int,
                        help='Number of epochs to run the algorithm')

    parser.add_argument('-bs',
                        '--batch-size',
                        type=int,
                        help='Size of batch')

    parser.add_argument('-nf',
                        '--num-folds',
                        type=int,
                        help='Number of cross validation folds to run')

    return parser


def main():
    parser = create_argparse()
    user_args = vars(parser.parse_args())

    train_file = user_args['train_file']
    test_file = user_args['test_file']

    print('Creating datasets ...')
    dataset = Dataset(train_file, test_file)
    train_data, test_data, train_targets, test_id = dataset.create_dataset()

    print('Creating model ...')
    categorical_columns = dataset.categorical_cols
    numeric_columns = dataset.numeric_cols
    bucket_columns = dataset.bucket_cols

    num_epochs = user_args['num_epochs']
    batch_size = user_args['batch_size']
    num_folds = user_args['num_folds']

    linear_model = LinearRegressionEstimator(
        train_data=train_data,
        train_targets=train_targets,
        test_data=test_data,
        numeric_columns=numeric_columns,
        bucket_columns=bucket_columns,
        categorical_columns=categorical_columns,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_folds=num_folds
    )

    print('Evaluating model ...')
    linear_model.evaluate()

    print('Creating prediction ...')
    pred = linear_model.run()
    final_predictions = np.exp(pred)

    submission = pd.DataFrame()
    submission['Id'] = test_id
    submission['SalePrice'] = final_predictions
    submission.head()

    submission.to_csv('submission1.csv', index=False)


if __name__ == '__main__':
    main()
