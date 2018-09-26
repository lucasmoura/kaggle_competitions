import numpy as np

from models.base_pipeline import (BaseFillMissing, BaseTransformations,
                                  BaseCreate, BaseDrop,
                                  BaseTargetTransform,
                                  BaseFinalize,
                                  BasePredictionTransform)


class RFFillMissing(BaseFillMissing):
    pass


class RFTransformations(BaseTransformations):
    def transform_target(self):
        pass


class RFCreate(BaseCreate):
    def __init__(self):
        super().__init__()
        self.category_columns.remove('GarageYrBlt')
        self.category_columns.append('MoSold')


class RFDrop(BaseDrop):
    def __init__(self):
        super().__init__()

        self.drop_columns.extend([
            'GarageYrBlt', 'TotRmsAbvGrd'
        ])


class RFTargetTransform(BaseTargetTransform):
    def transform_target(dataset):
        return dataset


class RFFinalize(BaseFinalize):
    pass


class RFPredictionTransform(BasePredictionTransform):

    def transform_predictions(predictions):
        return np.log(predictions)

    def revert_transform_predictions(predictions):
        return predictions
