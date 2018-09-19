import numpy as np

from manager.search_model import ModelSearcher, PipelineSearcher, MetricSearcher
from manager.submission import generate_submission
from preprocessing.pipeline import Pipeline
from utils.json import load_json
from utils.path import create_path


class ModelRunner:
    def __init__(self, model_name, train, target, test, num_folds, **kwargs):
        model, self.model_path = self.load_model(model_name)
        metric = self.load_metric()

        self.train, self.target, self.test = train, target, test
        self.num_folds = num_folds
        self.curr_folder = None
        self.model_name = model_name

        self.build_model(model, metric)

        super().__init__(**kwargs)

    def load_metric(self):
        return MetricSearcher('models').get_class('metric')

    def load_model(self, model_name):
        model_searcher = ModelSearcher('models')
        return model_searcher.get_class(model_name), model_searcher.path

    def extract_test_set(self):
        if self.test is not None:
            return self.test.copy()

        return None

    def extract_validation_set(self, column_name='fold'):
        train_data = self.train.copy()

        if self.curr_folder == -1:
            return train_data, None

        validation = train_data.loc[train_data[column_name] == self.curr_folder]
        train = train_data.loc[train_data[column_name] != self.curr_folder]

        return train, validation

    def extract_target_set(self, train=True):
        target = self.target.copy()

        if self.curr_folder == -1:
            target_y = target.loc[:, self.target.columns != 'fold']

        if train:
            target_y = target.loc[target.fold != self.curr_folder]
        else:
            target_y = target.loc[target.fold == self.curr_folder]

        target_y = target_y.loc[:, self.target.columns != 'fold']

        return self.apply_target_transformations(target_y)

    def apply_prediction_transformations(self, predictions):
        raise NotImplementedError

    def apply_target_transformations(self, target):
        raise NotImplementedError

    def extract_train_set(self):
        raise NotImplementedError

    def get_validation_predictions(self, validation_x):
        return self.ml_model.predict(validation_x)

    def build_model(self, model, metric):
        self.ml_model = model()
        self.ml_model.create_model()
        self.ml_model.set_evaluation_metric(metric)

    def evaluate_model(self, verbose):
        if verbose:
            print('Evaluating model')

        validation_x = self.pipeline.validation_data
        validation_y = self.extract_target_set(train=False).values.ravel()
        pred = self.get_validation_predictions(validation_x)

        trans_pred = self.apply_prediction_transformations(pred)
        trans_val_y = self.apply_prediction_transformations(
            validation_y).ravel()

        return self.ml_model.evaluate(trans_pred, trans_val_y)

    def perform_training(self, verbose):
        if verbose:
            print('Training model')

        train_x = self.extract_train_set()
        train_y = self.extract_target_set().values.ravel()

        self.ml_model.fit(train_x, train_y)

    def run(self, verbose=True):
        self.metric_values = []

        for folder_num in range(0, self.num_folds):
            self.curr_folder = folder_num
            self.perform_training(verbose)

            model_result = self.evaluate_model(verbose)
            self.metric_values.append(model_result)

            if verbose:
                print()

        mean_metric = np.mean(self.metric_values)
        print('Mean metric value: {}'.format(mean_metric))

        return mean_metric


class PipelineManager:

    def __init__(self, pipeline_name, **kwargs):
        super().__init__(**kwargs)
        self.pipeline_name = pipeline_name

    def load_pipeline(self, model_path):
        pipeline_searcher = PipelineSearcher(model_path)
        operations = pipeline_searcher.get_class(self.pipeline_name)
        return operations, pipeline_searcher.path

    def get_operations(self, model_path):
        operations, self.save_path = self.load_pipeline(model_path)

        return operations

    def build_pipeline(self, model_path):
        self.pipeline = Pipeline()
        operations = self.get_operations(model_path)

        (*pipeline_ops, self.target_transform,
         finalize, self.pred_transformer) = operations

        self.pipeline.set_operations(*pipeline_ops)
        self.pipeline.set_finalize(finalize)

    def set_pipeline(self, train, validation, test):
        self.pipeline.set_dataset(train, validation, test)

    def run_pipeline(self, verbose):
        if verbose:
            print('Running pipeline')

        self.pipeline.run_pipeline(verbose)


class ModelEvaluation(PipelineManager, ModelRunner):
    def __init__(self, train, target, test, model_name, pipeline_name,
                 num_folds, create_submission, id_column, target_column):

        super().__init__(
            train=train,
            target=target,
            test=test,
            model_name=model_name,
            pipeline_name=pipeline_name,
            num_folds=num_folds)

        self.create_submission = create_submission
        self.id_column = id_column
        self.target_column = target_column

        self.build_pipeline(self.model_path)
        self.set_model_config()
        self.predictions = []

    def apply_target_transformations(self, target):
        return self.target_transform.transform_target(target)

    def apply_prediction_transformations(self, predictions):
        return self.pred_transformer.transform_predictions(
            predictions)

    def extract_train_set(self):
        return self.pipeline.train_data

    def set_model_config(self):
        config_path = create_path(self.save_path, 'config.json')
        config = load_json(config_path)

        self.ml_model.set_config(config)

    def get_test_predictions(self):
        test_x = self.pipeline.test_data
        return self.pred_transformer.revert_transform_predictions(
                self.ml_model.predict(test_x))

    def perform_training(self, verbose=True):
        train, validation = self.extract_validation_set()
        test = self.extract_test_set()

        self.set_pipeline(train, validation, test)
        self.run_pipeline(verbose)

        super().perform_training(verbose)

    def run(self, verbose=True):
        metric_result = super().run(verbose)

        if verbose:
            print()

        if self.create_submission:
            if verbose:
                print('Training with full dataset')

            self.curr_folder = -1
            self.perform_training(verbose=verbose)
            predictions = self.get_test_predictions()

            generate_submission(
                predictions, self.id_column, self.target_column,
                self.test[self.id_column], self.save_path
            )

        return metric_result
