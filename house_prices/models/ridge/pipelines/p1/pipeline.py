from models.base_pipeline import (BaseFillMissing, BaseTransformations,
                                  BaseCreate, BaseDrop, BaseTargetTransform,
                                  BaseFinalize, BasePredictionTransform)


class RFillMissing(BaseFillMissing):
    pass


class RTransformations(BaseTransformations):
    pass


class RCreate(BaseCreate):
    def __init__(self):
        super().__init__()
        self.category_columns.remove('GarageYrBlt')


class RDrop(BaseDrop):
    def __init__(self):
        super().__init__()

        self.drop_columns.append('GarageYrBlt')


class RTargetTransform(BaseTargetTransform):
    pass


class RFinalize(BaseFinalize):
    pass


class RPredictionTransform(BasePredictionTransform):
    pass
