from models.base_pipeline import (BaseFillMissing, BaseTransformations,
                                  BaseCreate, BaseDrop, BaseTargetTransform,
                                  BaseFinalize, BasePredictionTransform)


class LFillMissing(BaseFillMissing):
    pass


class LTransformations(BaseTransformations):
    pass


class LCreate(BaseCreate):
    def __init__(self):
        super().__init__()
        self.category_columns.remove('GarageYrBlt')


class LDrop(BaseDrop):
    def __init__(self):
        super().__init__()

        self.drop_columns.extend([
            'GarageYrBlt', 'TotRmsAbvGrd'
        ])


class LTargetTransform(BaseTargetTransform):
    pass


class LFinalize(BaseFinalize):
    pass


class LPredictionTransform(BasePredictionTransform):
    pass
