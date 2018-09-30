from scipy.stats import randint

from kaggleflow.utils.json import load_json


class TuningParser:

    def __init__(self, tuning_file):
        self.tuning_json = load_json(tuning_file)

    def parse_json(self):
        tuning_variables = dict()
        for variable, parameters in self.tuning_json.items():
            tuning_variables[variable] = self.create_distribution(parameters)

        return tuning_variables

    def create_distribution(self, parameters):
        if parameters['distribution'] == 'uniform':
            return self.create_uniform_distribution(parameters)

    def create_uniform_distribution(self, parameters):
        if 'values' in parameters:
            return parameters['values']

        return randint(parameters['min'], parameters['max'])
