import numpy as np
from models.base_pipeline import (BaseFillMissing, BaseTransformations,
                                  BaseCreate, BaseDrop, BaseTargetTransform,
                                  BaseFinalize, BasePredictionTransform)


class LFillMissing(BaseFillMissing):
    pass


class LTransformations(BaseTransformations):
    def transform_to_ordinal(self):
        super().transform_to_ordinal()

        columns = ['LandSlope']
        ordinal_map = {'NP': 0, 'Sev': 1, 'Mod': 2, 'Gtl': 3}
        self.apply_ordinal_transformation(ordinal_map, columns)

        columns = ['LotShape']
        ordinal_map = {'NP': 0, 'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4}
        self.apply_ordinal_transformation(ordinal_map, columns)

        columns = ['PavedDrive']
        ordinal_map = {'NP': 0, 'N': 1, 'P': 2, 'Y': 3}
        self.apply_ordinal_transformation(ordinal_map, columns)

        columns = ['Functional']
        ordinal_map = {'NP': 0, 'Sal': 1, 'Sev': 2, 'Maj2': 3,
                       'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7,
                       'Typ': 8}
        self.apply_ordinal_transformation(ordinal_map, columns)

        columns = ['GarageFinish']
        ordinal_map = {'NP': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
        self.apply_ordinal_transformation(ordinal_map, columns)

        columns = ['CentralAir']
        ordinal_map = {'NP': 0, 'N': 1, 'Y': 2}
        self.apply_ordinal_transformation(ordinal_map, columns)

        columns = ['MSSubClass']
        ordinal_map = {'NP': 0, 20: 1, 30: 2, 40: 3, 45: 4,
                       50: 5, 60: 6, 70: 7, 75: 8, 80: 8,
                       85: 9, 90: 9, 120: 10, 150: 11,
                       160: 12, 180: 13, 190: 14}
        self.apply_ordinal_transformation(ordinal_map, columns)


class LCreate(BaseCreate):
    def __init__(self):
        super().__init__()

        self.category_columns.remove('LandSlope')
        self.category_columns.remove('LotShape')
        self.category_columns.remove('PavedDrive')
        self.category_columns.remove('Functional')
        self.category_columns.remove('GarageFinish')
        self.category_columns.remove('GarageType')
        self.category_columns.remove('CentralAir')
        self.category_columns.remove('MSSubClass')
        self.category_columns.remove('MasVnrType')

    def create_all_sf(self):
        for dataset in self.loop_datasets():
            dataset['all_sf'] = np.log1p(
                dataset['GrLivArea'] * dataset['TotalBsmtSF'])


class LDrop(BaseDrop):
    def __init__(self):
        super().__init__()

        self.drop_columns.append('TotRmsAbvGrd')
        self.drop_columns.append('Fence')
        self.drop_columns.append('FireplaceQu')
        self.drop_columns.append('GarageFinish')
        self.drop_columns.append('GarageType')
        self.drop_columns.append('GarageCond')
        self.drop_columns.append('MasVnrArea')
        self.drop_columns.append('MasVnrType')
        self.drop_columns.append('BsmtFinType1')
        self.drop_columns.append('BsmtFinType2')
        self.drop_columns.append('BsmtCond')


class LTargetTransform(BaseTargetTransform):
    pass


class LFinalize(BaseFinalize):
    pass


class LPredictionTransform(BasePredictionTransform):
    pass
