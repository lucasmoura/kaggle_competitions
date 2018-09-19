from sklearn.model_selection import ParameterSampler

from manager.model_manager import ModelEvaluation
from manager.tuning.parser import TuningParser
from utils.path import create_path
from utils.json import save_json


class ModelTuning(ModelEvaluation):

    def __init__(self, train, target, model_name, pipeline_name, num_folds, num_iter):
        super().__init__(
            train=train,
            target=target,
            test=None,
            model_name=model_name,
            pipeline_name=pipeline_name,
            num_folds=num_folds,
            create_submission=False,
            id_column=None,
            target_column=None
        )

        self.num_iter = num_iter

    def get_variables_distribution(self):
        tuning_json_path = create_path(self.save_path, 'tuning.json')

        parser = TuningParser(tuning_json_path)
        return parser.parse_json()

    def parameter_sampler(self):
        variables_distribution = self.get_variables_distribution()
        return ParameterSampler(
            param_distributions=variables_distribution,
            n_iter=self.num_iter,
            random_state=None)

    def save_best_config(self, best_config):
        save_path = create_path(self.save_path, 'best_config.json')
        save_json(save_path, best_config)

    def run(self, verbose=True):
        best_metric_value = 10000
        best_config = None

        for parameter in self.parameter_sampler():
            self.ml_model.set_config(parameter)

            if verbose:
                print('Running new round of hyperparameter tunning ...')

            mean_metric = super().run(verbose)

            if mean_metric < best_metric_value:
                best_metric_value = mean_metric
                best_config = parameter

            if verbose:
                print('------------\n')

        if verbose:
            print('Best metric value found: {}'.format(best_metric_value))

            self.save_best_config(best_config)
