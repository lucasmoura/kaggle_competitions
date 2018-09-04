from manager.search_model import ModelSearcher, PipelineSearcher, MetricSearcher
from preprocessing.pipeline import Pipeline


class ModelManager:

    def __init__(self, train, test, model_name, pipeline_name, num_folds):
        self.train, self.test = train, test
        model, operations, metric = self.load_modules(model_name, pipeline_name)

        self.build_model(model, metric)
        self.build_pipeline(operations)

        self.num_folds = num_folds

    def load_modules(self, model_name, pipeline_name):
        model_searcher = ModelSearcher('models')
        model = model_searcher.get_class(model_name)

        pipeline_searcher = PipelineSearcher(model_searcher.path)
        operations = pipeline_searcher.get_class(pipeline_name)

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

        validation = train_data.loc[train_data[column_name] == folder_num]
        train = train_data.loc[train_data[column_name] != folder_num]

        return train, validation

    def update_model(self):
        raise NotImplementedError

    def update_results(self, model_result):
        raise NotImplementedError

    def run(self, verbose=True):
        self.metric_values = []

        for folder_num in range(0, self.num_folds):
            self.set_pipeline(folder_num)

            if verbose:
                print('Running pipeline')

            self.pipeline.run_pipeline(verbose)
            self.update_model()

            if verbose:
                print('Gathering data')

            train_x, train_y = self.pipeline.train_data
            validation_x, validation_y = self.pipeline.validation_data

            if verbose:
                print('Training model')

            self.ml_model.fit(train_x, train_y)

            if verbose:
                print('Evaluating model')

            model_result = self.ml_model.evaluate(validation_x, validation_y)

            self.update_results(model_result)
            print()


class ModelEvaluation(ModelManager):
    def update_model(self):
        pass

    def update_results(self, model_result):
        self.metric_values.append(model_result)
