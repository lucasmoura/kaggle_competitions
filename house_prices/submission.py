import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from preprocessing.dataset import Dataset
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

    parser.add_argument('-uv',
                        '--use-validation',
                        type=int,
                        help='If validation data should be used')

    return parser


def main():
    parser = create_argparse()
    user_args = vars(parser.parse_args())

    train_file = user_args['train_file']
    test_file = user_args['test_file']
    use_validation = True if user_args['use_validation'] == 1 else False

    print('Creating datasets ...')
    dataset = Dataset(train_file, test_file)
    train_data, test_data, train_targets, test_id = dataset.create_dataset()

    if use_validation:
        print('Creating validation dataset ...')
        (train_data, validation_data,
         train_targets, validation_targets) = train_test_split(
            train_data, train_targets, random_state=42, test_size=.2)
    else:
        validation_data, validation_targets = None, None

    print('Creating model ...')
    categorical_columns = dataset.categorical_cols
    numeric_columns = dataset.numeric_cols
    bucket_columns = dataset.bucket_cols

    num_epochs = user_args['num_epochs']
    batch_size = user_args['batch_size']

    linear_model = LinearRegressionEstimator(
        train_data=train_data,
        train_targets=train_targets,
        use_validation=use_validation,
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
    submission['Id'] = test_id
    submission['SalePrice'] = final_predictions
    submission.head()

    submission.to_csv('submission1.csv', index=False)


if __name__ == '__main__':
    main()
