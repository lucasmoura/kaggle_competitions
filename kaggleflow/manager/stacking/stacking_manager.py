import pandas as pd

from kaggleflow.manager.model_manager import ModelEvaluation, ModelRunner
from kaggleflow.manager.submission import generate_submission
from kaggleflow.preprocessing.load import load_dataset
from kaggleflow.manager.search_model import StackingSearcher
from kaggleflow.utils.path import create_path
from kaggleflow.utils.json import load_json


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

    def generate_prediction_column_name(self, name):
        return name + self.model_name + '_' + self.pipeline_name

    def create_stack_pred(self):
        stack_df = pd.DataFrame.from_records(self.predictions)
        prediction_column = self.generate_prediction_column_name('Prediction')
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
        self.target_column = self.generate_prediction_column_name(
            self.target_column)
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

        self.stack_transformer = self.load_stacking_transformation()

    def load_stacking_transformation(self):
        searcher = StackingSearcher('models')
        return searcher.get_class('stacking')

    def extract_set(self, dataset, column_name):
        return dataset.loc[:, ~(dataset.columns == column_name)]

    def extract_train_set(self):
        train_data = self.extract_set(self.train_data, 'Fold')
        return self.stack_transformer.transform(train_data)

    def extract_validation_set(self):
        validation_data = self.extract_set(self.validation_data, 'Fold')
        return self.stack_transformer.transform(validation_data)

    def extract_test_set(self):
        test_data = self.extract_set(self.test, self.id_column)
        return self.stack_transformer.transform(test_data)

    def apply_target_transformations(self, target):
        return self.stack_transformer.transform(target)

    def apply_prediction_transformations(self, predictions):
        return self.stack_transformer.prediction_transform(predictions)

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
        return self.stack_transformer.revert(
            self.ml_model.predict(test_x))

    def run(self, verbose=True):
        super().run()

        self.curr_folder = -1
        self.perform_training(verbose=verbose)
        predictions = self.get_test_predictions()

        generate_submission(
            predictions, self.id_column, self.target_column,
            self.test[self.id_column], self.model_path
        )
