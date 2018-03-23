import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from preprocessing.dataset import TrainDataset, TestDataset
from model.estimator import LinearRegressionEstimator


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

    return parser


def main():
    parser = create_argparse()
    user_args = vars(parser.parse_args())

    train_file = user_args['train_file']
    test_file = user_args['test_file']

    print('Creating train dataset ...')
    train_dataset = TrainDataset(train_file)
    train_data = train_dataset.create_dataset()
    train_targets = train_dataset.targets

    print('Creating test dataset ...')
    test_data = TestDataset(test_file, train=False).create_dataset()

    print('Creating validation dataset ...')
    (train_data, validation_data,
     train_targets, validation_targets) = train_test_split(
        train_data, train_targets, random_state=42, test_size=.2)

    print('Creating model ...')

    numeric_columns = train_data.columns
    bucket_columns, categorical_columns = {}, {}

    num_epochs = user_args['num_epochs']
    batch_size = user_args['batch_size']

    linear_model = LinearRegressionEstimator(
        train_data=train_data,
        train_targets=train_targets,
        validation_data=validation_data,
        validation_targets=validation_targets,
        test_data=test_data,
        numeric_columns=numeric_columns,
        bucket_columns=bucket_columns,
        categorical_columns=categorical_columns,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    pred = linear_model.run()
    final_predictions = np.exp(pred)

    submission = pd.DataFrame()
    submission['Id'] = test_data.Id
    submission['SalePrice'] = final_predictions
    submission.head()
    
    submission.to_csv('submission1.csv', index=False)


if __name__ == '__main__':
    main()
