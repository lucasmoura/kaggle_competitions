import pandas as pd

from manager.search_model import ModelSearcher, PipelineSearcher, MetricSearcher
from preprocessing.pipeline import Pipeline


class ModelManager:

    def __init__(self, train, test, model_name, pipeline_name,
                 num_folds, create_submission):
        self.train, self.test = train, test
        model, operations, metric = self.load_modules(model_name, pipeline_name)

        self.build_model(model, metric)
        self.build_pipeline(operations)

        self.num_folds = num_folds
        self.create_submission = create_submission

    def load_modules(self, model_name, pipeline_name):
        model_searcher = ModelSearcher('models')
        model = model_searcher.get_class(model_name)

        pipeline_searcher = PipelineSearcher(model_searcher.path)
        operations = pipeline_searcher.get_class(pipeline_name)
        self.save_path = pipeline_searcher.path

        metric = MetricSearcher('models').get_class('metric')

        return (model, operations, metric)

    def build_model(self, model, metric):
        self.ml_model = model()
        self.ml_model.create_model()
        self.ml_model.set_evaluation_metric(metric)

    def build_pipeline(self, operations):
        self.pipeline = Pipeline()

        *pipeline_ops, finalize = operations
        self.pipeline.set_operations(*pipeline_ops)
        self.pipeline.set_finalize(finalize)

    def set_pipeline(self, folder_num):
        train, validation = self.extract_validation_set(folder_num)
        test = self.test.copy()

        self.pipeline.set_dataset(train, validation, test)

    def extract_validation_set(self, folder_num, column_name='fold'):
        train_data = self.train.copy()

        if folder_num == -1:
            return train_data, None

        validation = train_data.loc[train_data[column_name] == folder_num]
        train = train_data.loc[train_data[column_name] != folder_num]

        return train, validation

    def update_model(self):
        raise NotImplementedError

    def update_results(self, model_result):
        raise NotImplementedError

    def run_pipeline(self, verbose):
        if verbose:
            print('Running pipeline')

        self.pipeline.run_pipeline(verbose)

    def fit_model(self, verbose):
        if verbose:
            print('Training model')

        train_x, train_y = self.pipeline.train_data
        self.ml_model.fit(train_x, train_y)

    def evaluate_model(self, verbose):
        if verbose:
            print('Evaluating model')

        validation_x, validation_y = self.pipeline.validation_data
        return self.ml_model.evaluate(validation_x, validation_y)

    def perform_training(self, folder_num, verbose=True):
        self.set_pipeline(folder_num)
        self.run_pipeline(verbose)
        self.update_model()

        self.fit_model(verbose)

    def run(self, verbose=True):
        self.metric_values = []

        for folder_num in range(0, self.num_folds):
            self.perform_training(folder_num, verbose)
            model_result = self.evaluate_model(verbose)
            self.update_results(model_result)

            if verbose:
                print()


class ModelEvaluation(ModelManager):
    def create_submission_path(self):
        save_folder = self.save_path.replace('.', '/')
        return save_folder + '/submission.csv'

    def generate_submission(self, predictions, verbose):
        if verbose:
            print('Creating submission')

        submission_df = pd.DataFrame({'Id': self.test.Id, 'SalePrice': predictions})
        submission_df.to_csv(self.create_submission_path(), index=False)

    def update_model(self):
        pass

    def update_results(self, model_result):
        self.metric_values.append(model_result)

    def get_test_predictions(self):
        test_x = self.pipeline.test_data
        return self.ml_model.predict(test_x)

    def run(self, verbose=True):
        super().run(verbose)

        if verbose:
            print()

        if self.create_submission:
            if verbose:
                print('Training with full dataset')

            self.perform_training(folder_num=-1, verbose=verbose)
            predictions = self.get_test_predictions()
            self.generate_submission(predictions, verbose)
