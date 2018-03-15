import math
import tempfile

import tensorflow as tf

from sklearn.metrics import mean_squared_error

from model.input_pipeline import train_input_fn, test_input_fn


class LinearRegressionEstimator:

    def __init__(self, train_data, train_targets, validation_data,
                 validation_targets, numeric_columns, test_data,
                 bucket_columns, categorical_columns, num_epochs, batch_size):
        self.train_data = train_data
        self.train_targets = train_targets

        self.validation_data = validation_data
        self.validation_targets = validation_targets

        self.test_data = test_data

        self.numeric_columns = numeric_columns
        self.bucket_columns = bucket_columns
        self.categorical_columns = categorical_columns

        self.num_epochs = num_epochs
        self.batch_size = batch_size

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
            column = tf.feature_column.categorical_column_with_identity(
                key=key,
                num_buckets=value
            )

            categorical_columns.append(column)

        return categorical_columns

    def get_feature_columns(self):
        categorical_columns = self.get_categorical_columns()
        bucket_columns = self.get_bucketized_columns()
        numeric_columns = self.get_numeric_columns()

        return categorical_columns + bucket_columns + numeric_columns

    def run(self):
        columns = self.get_feature_columns()

        model_dir = tempfile.mkdtemp()
        self.estimator = tf.estimator.LinearRegressor(
            model_dir=model_dir, feature_columns=columns
        )

        self.estimator.train(
             input_fn=train_input_fn(
                 data_dataframe=self.train_data,
                 target_dataframe=self.train_targets,
                 batch_size=self.batch_size,
                 num_epochs=self.num_epochs,
                 should_shuffle=True
             )
         )

        predictions = self.estimator.predict(
             input_fn=test_input_fn(
                 data_dataframe=self.validation_data,
             )
         )

        pred = [x['predictions'].item(0) for x in predictions]
        val_values = self.validation_targets.values

        rmse = mean_squared_error(pred, val_values)
        print(rmse)

