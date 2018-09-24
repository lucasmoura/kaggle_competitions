
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

        self.category_columns = [
            'MSSubClass', 'MSZoning', 'LotShape', 'LandContour', 'LotConfig',
            'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
            'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
            'MasVnrType', 'Foundation', 'CentralAir', 'Electrical',
            'Functional', 'GarageType',
            'GarageFinish', 'PavedDrive', 'SaleType', 'SaleCondition'
        ]


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
