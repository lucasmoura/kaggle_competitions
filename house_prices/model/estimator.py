import math
import tempfile

import tensorflow as tf

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from model.input_pipeline import train_input_fn, test_input_fn


class LinearRegressionEstimator:

    def __init__(self, train_data, train_targets,
                 numeric_columns, test_data, bucket_columns,
                 categorical_columns, num_epochs, batch_size,
                 num_folds):
        self.train_data = train_data
        self.train_targets = train_targets
        self.test_data = test_data

        self.numeric_columns = numeric_columns
        self.bucket_columns = bucket_columns
        self.categorical_columns = categorical_columns

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_folds = num_folds

    def get_numeric_columns(self):
        columns = []
        for num_column in self.numeric_columns:
            columns.append(
                tf.feature_column.numeric_column(
                    key=num_column
                )
            )

        return columns

    def get_bucketized_columns(self):
        bucket_columns = []
        for key, value in self.bucket_columns.items():
            numeric = tf.feature_column.numeric_column(
                key=key
            )

            bucket = tf.feature_column.bucketized_column(
                numeric,
                boundaries=value
            )

            bucket_columns.append(bucket)

        return bucket_columns

    def get_categorical_columns(self):
        categorical_columns = []

        for key, value in self.categorical_columns.items():
            column = tf.feature_column.categorical_column_with_vocabulary_list(
                key=key,
                vocabulary_list=value
            )

            categorical_columns.append(column)

        return categorical_columns

    def get_feature_columns(self):
        categorical_columns = self.get_categorical_columns()
        bucket_columns = self.get_bucketized_columns()
        numeric_columns = self.get_numeric_columns()

        return categorical_columns + bucket_columns + numeric_columns

    def evaluate(self):
        k_fold = KFold(n_splits=self.num_folds)
        rmse = []
        index = 1

        for train_index, validation_index in k_fold.split(self.train_data):
            print('Running fold {}'.format(index))
            index += 1

            train_data = self.train_data.loc[train_index, :]
            train_targets = self.train_targets.loc[train_index, :]

            validation_data = self.train_data.loc[validation_index, :]
            validation_targets = self.train_targets.loc[validation_index, :]

            columns = self.get_feature_columns()
            estimator = self.create_model(columns)
            self.train_model(estimator, train_data, train_targets)
            predictions = self.model_predictions(estimator, validation_data)

            pred = [x['predictions'].item(0) for x in predictions]
            val_values = validation_targets.values

            rmse.append(math.sqrt(mean_squared_error(pred, val_values)))

        rmse_value = sum(rmse) / len(rmse)
        print('Rmse: {}'.format(rmse_value))

    def create_model(self, columns):
        model_dir = tempfile.mkdtemp()
        estimator = tf.estimator.LinearRegressor(
            model_dir=model_dir, feature_columns=columns,
            optimizer=tf.train.FtrlOptimizer(
                    learning_rate=0.1,
                    l1_regularization_strength=1.0,
                    l2_regularization_strength=1.0)
        )

        return estimator

    def train_model(self, estimator, train_data, train_targets):
        estimator.train(
            input_fn=train_input_fn(
                data_dataframe=train_data,
                target_dataframe=train_targets,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                should_shuffle=True
            )
        )

    def model_predictions(self, estimator, data_dataframe):
        predictions = estimator.predict(
            input_fn=test_input_fn(
                data_dataframe=data_dataframe
            )
        )

        return predictions

    def run(self):
        columns = self.get_feature_columns()
        estimator = self.create_model(columns)
        self.train_model(estimator, self.train_data, self.train_targets)
        predictions = self.model_predictions(estimator, self.test_data)

        pred = [x['predictions'].item(0) for x in predictions]
        return pred
