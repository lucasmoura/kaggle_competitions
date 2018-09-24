from preprocessing.transform import transform_to_log1_scale
from models.base_pipeline import (BaseFillMissing, BaseTransformations,
                                  BaseCreate, BaseDrop, BaseTargetTransform,
                                  BaseFinalize, BasePredictionTransform)


class RFillMissing(BaseFillMissing):
    pass


class RTransformations(BaseTransformations):

    def __init__(self):
        super().__init__()

        self.numeric_columns = [
            'LotFrontage', 'LotArea', 'MasVnrArea',
            'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
            '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
            'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
            '3SsnPorch', 'ScreenPorch'
        ]

    def transform_numeric_data(self):
        for column in self.numeric_columns:
            for dataset in self.loop_datasets():
                transform_to_log1_scale(dataset, column)


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
