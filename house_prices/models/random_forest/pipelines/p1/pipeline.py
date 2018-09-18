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
    pass


class RFDrop(BaseDrop):
    pass


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
