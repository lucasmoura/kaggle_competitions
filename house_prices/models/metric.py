from math import sqrt

from sklearn.metrics import mean_squared_error

from manager.base_metric import Metric


class HouseMetric(Metric):

    def compute(predictions, targets):
        return sqrt(mean_squared_error(predictions, targets))
