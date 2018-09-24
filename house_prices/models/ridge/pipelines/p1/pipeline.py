from models.base_pipeline import (BaseFillMissing, BaseTransformations,
                                  BaseCreate, BaseDrop, BaseTargetTransform,
                                  BaseFinalize, BasePredictionTransform)


class RFillMissing(BaseFillMissing):
    pass


class RTransformations(BaseTransformations):
    pass


class RCreate(BaseCreate):
    pass


class RDrop(BaseDrop):
    pass


class RTargetTransform(BaseTargetTransform):
    pass


class RFinalize(BaseFinalize):
    pass


class RPredictionTransform(BasePredictionTransform):
    pass
