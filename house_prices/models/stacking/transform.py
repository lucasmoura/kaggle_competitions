import numpy as np

from manager.stacking.stacking_transform import StackingTransform


class HouseStackingTransform(StackingTransform):

    @staticmethod
    def transform(values):
        return np.log(values)

    @staticmethod
    def revert(values):
        return np.exp(values)

    @staticmethod
    def prediction_transform(values):
        return values
