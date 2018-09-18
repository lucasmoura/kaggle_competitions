from models.base_pipeline import (BaseFillMissing, BaseTransformations,
                                  BaseCreate, BaseDrop, BaseTargetTransform,
                                  BaseFinalize, BasePredictionTransform)


class LFillMissing(BaseFillMissing):
    pass


class LTransformations(BaseTransformations):
    pass


class LCreate(BaseCreate):
    pass


class LDrop(BaseDrop):
    pass


class LTargetTransform(BaseTargetTransform):
    pass


class LFinalize(BaseFinalize):
    pass


class LPredictionTransform(BasePredictionTransform):
    pass
