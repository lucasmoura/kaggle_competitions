import pandas as pd
import numpy as np

from manager.model_manager import ModelEvaluation, ModelRunner
from manager.submission import generate_submission
from preprocessing.load import load_dataset
from utils.path import create_path
from utils.json import load_json


class StackingModel(ModelEvaluation):

    def __init__(self, train, target, test, model_name, pipeline_name,
                 num_folds, create_submission, id_column, target_column):
        super().__init__(
            train=train,
            target=target,
            test=test,
            model_name=model_name,
            pipeline_name=pipeline_name,
            num_folds=num_folds,
            create_submission=create_submission,
            id_column=id_column,
            target_column=target_column
        )

        self.predictions = []

    def create_stack_pred(self):
        stack_df = pd.DataFrame.from_records(self.predictions)
        prediction_column = 'Prediction_' + self.model_name
        stack_df.set_axis(['Fold', prediction_column], axis=1, inplace=True)
        stack_df.to_csv(
            create_path(self.save_path, 'stack.csv'),
            index=False
        )

    def get_validation_predictions(self, validation_x):
        predictions = super().get_validation_predictions(validation_x)
        updated_predictions = self.pred_transformer.revert_transform_predictions(
            predictions)

        pred_list = [(self.curr_folder, pred) for pred in updated_predictions]
        self.predictions.extend(pred_list)

        return predictions

    def save_submission(self, predictions):
        self.target_column = self.target_column + '_' + self.model_name
        super().save_submission(predictions)

    def run(self, verbose=True):
        super().run(verbose)
        self.create_stack_pred()


class StackingEvaluation(ModelRunner):
    def __init__(self, target, stacking_file, num_folds,
                 id_column, target_column):
        stacking_dict = load_json(stacking_file)

        train = self.create_train_stacking_data(stacking_dict)
        test = self.create_test_stacking_data(stacking_dict)

        super().__init__(
            train=train,
            target=target,
            test=test,
            model_name='stacking',
            num_folds=num_folds)

        self.id_column = id_column
        self.target_column = target_column

        self.set_model_config(self.model_path)

    def extract_set(self, dataset, column_name):
        return dataset.loc[:, ~(dataset.columns == column_name)]

    def extract_train_set(self):
        return self.extract_set(self.train_data, 'Fold')

    def extract_validation_set(self):
        return self.extract_set(self.validation_data, 'Fold')

    def extract_test_set(self):
        return self.extract_set(self.test, self.id_column)

    def apply_target_transformations(self, target):
        return target

    def apply_prediction_transformations(self, predictions):
        return np.log(predictions)

    def perform_training(self, verbose=True):
        self.train_data, self.validation_data = self.parse_train_set(
            column_name='Fold')
        super().perform_training(verbose=True)

    def create_model_path(self, model, pipeline_name, file_name):
        return 'models/' + model + '/pipelines/' + pipeline_name + '/' + file_name

    def get_stacking_csvs(self, stacking_dict, file_name):
        stacking_csvs = []

        for model, pipelines in stacking_dict.items():
            for pipeline in pipelines:
                stacking_csvs.append(
                    self.create_model_path(
                        model, pipeline, file_name))

        return stacking_csvs

    def get_stacking_data(self, stacking_csv):
        pandas_csv = (load_dataset(csv) for csv in stacking_csv)
        single_df = pd.concat(pandas_csv, axis=1)

        return single_df.loc[:, ~single_df.columns.duplicated()]

    def create_train_stacking_data(self, stacking_dict):
        return self.create_stacking_data(stacking_dict, 'stack.csv')

    def create_test_stacking_data(self, stacking_dict):
        return self.create_stacking_data(stacking_dict, 'submission.csv')

    def create_stacking_data(self, stacking_dict, file_name):
        stacking_csvs = self.get_stacking_csvs(stacking_dict, file_name)
        return self.get_stacking_data(stacking_csvs)

    def get_test_predictions(self):
        test_x = self.extract_test_set()
        return self.ml_model.predict(test_x)

    def run(self, verbose=True):
        super().run()

        self.curr_folder = -1
        self.perform_training(verbose=verbose)
        predictions = self.get_test_predictions()

        generate_submission(
            predictions, self.id_column, self.target_column,
            self.test[self.id_column], self.model_path
        )
