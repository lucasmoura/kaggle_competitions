import numpy as np

from models.base_pipeline import (BaseFillMissing, BaseTransformations,
                                  BaseCreate, BaseDrop, BaseFinalize,
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


class RFFinalize(BaseFinalize):
    pass


class RFPredictionTransform(BasePredictionTransform):

    def transform_predictions(predictions):
        return np.log(predictions)

    def revert_transform_predictions(predictions):
        return predictions
