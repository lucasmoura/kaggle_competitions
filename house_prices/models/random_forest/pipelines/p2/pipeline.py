
from models.base_pipeline import (BaseFillMissing, BaseTransformations,
                                  BaseCreate, BaseDrop, BaseTargetTransform,
                                  BaseFinalize, BasePredictionTransform)


class RFFillMissing(BaseFillMissing):
    pass


class RFTransformations(BaseTransformations):
    pass


class RFCreate(BaseCreate):
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


class RFDrop(BaseDrop):
    def __init__(self):
        super().__init__()

        self.drop_columns.extend([
            'GarageYrBlt', 'TotRmsAbvGrd'
        ])


class RFTargetTransform(BaseTargetTransform):
    pass


class RFFinalize(BaseFinalize):
    pass


class RFPredictionTransform(BasePredictionTransform):
    pass
