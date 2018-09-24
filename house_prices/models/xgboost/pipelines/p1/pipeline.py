import numpy as np

from models.base_pipeline import (BaseFillMissing, BaseTransformations,
                                  BaseCreate, BaseDrop,
                                  BaseTargetTransform,
                                  BaseFinalize,
                                  BasePredictionTransform)


class XBGFillMissing(BaseFillMissing):
    pass


class XBGTransformations(BaseTransformations):
    def transform_target(self):
        pass


class XBGCreate(BaseCreate):
    pass


class XBGDrop(BaseDrop):
    pass


class XBGTargetTransform(BaseTargetTransform):
    def transform_target(dataset):
        return dataset


class XBGFinalize(BaseFinalize):
    pass


class XBGPredictionTransform(BasePredictionTransform):

    def transform_predictions(predictions):
        return np.log(predictions)

    def revert_transform_predictions(predictions):
        return predictions
