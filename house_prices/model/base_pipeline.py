from preprocessing.pipeline import Transformations, FillMissing, Create, Drop
from preprocessing.transform import transform_categorical_column
from preprocessing.missing_data import fill_nan_with_value, drop_columns


class BaseCreate(Create):
    pass


class BaseFillMissing(FillMissing):

    def fill_na_categorical_columns(self):
        columns = ['BsmtQual', 'BsmtCond', 'FireplaceQu', 'GarageQual',
                   'GarageCond', 'GarageFinish', 'GarageType', 'GarageYrBlt',
                   'MasVnrType', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                   'Fence']

        for dataset in self.data:
            fill_nan_with_value(dataset, columns, 'NP')

    def fill_mas_vnr_type(self):
        for dataset in self.data:
            fill_nan_with_value(dataset, ['MasVnrType'], 'NP')

    def fill_with_zero(self):
        columns = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                   'TotalBsmtSF', 'GarageCars', 'GarageArea',
                   'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']

        for column in columns:
            for dataset in self.data:
                fill_nan_with_value(dataset, [column], 0)

    def fill_lot_frontage(self):
        train = self.data[0]
        mean = train['LotFrontage'].mean()

        for dataset in self.data:
            fill_nan_with_value(dataset, ['LotFrontage'], mean)

    def mode_fill(self, column):
        train = self.data[0]
        mode = train[column].mode()[0]

        for dataset in self.data:
            fill_nan_with_value(dataset, [column], mode)

    def fill_electrical(self):
        self.mode_fill('Electrical')

    def fill_ms_zoning(self):
        self.mode_fill('MSZoning')

    def fill_functional(self):
        self.mode_fill('Functional')

    def fill_sale_type(self):
        self.mode_fill('SaleType')

    def fill_exterior_1st(self):
        self.mode_fill('Exterior1st')

    def fill_exterior_2nd(self):
        self.mode_fill('Exterior2nd')

    def fill_kitchen_qual(self):
        self.mode_fill('KitchenQual')


class BaseTransformations(Transformations):

    def apply_ordinal_transformation(self, ordinal_map, columns):
        for column in columns:
            for dataset in self.data:
                transform_categorical_column(dataset, column, ordinal_map)

    def transform_to_ordinal(self):
        ordinal_map = {'NP': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        columns = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual']
        self.apply_ordinal_transformation(ordinal_map, columns)

        ordinal_map = {'NP': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
        columns = ['BsmtFinType1', 'BsmtFinType2']
        self.apply_ordinal_transformation(ordinal_map, columns)

        columns = ['BsmtExposure']
        ordinal_map = {'NP': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
        self.apply_ordinal_transformation(ordinal_map, columns)

        columns = ['Fence']
        ordinal_map = {'NP': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
        self.apply_ordinal_transformation(ordinal_map, columns)

    def transform_type_to_categorical(self):
        columns = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'LotConfig',
                   'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                   'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
                   'MasVnrType', 'Foundation', 'CentralAir', 'Electrical',
                   'Functional', 'GarageType', 'GarageYrBlt', 
                   'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 
                   'SaleType', 'SaleCondition']

        for column in columns:
            pass



class BaseDrop(Drop):

    def drop_columns_from_datasets(self):
        columns = ['PoolQC', 'MiscFeature', 'Alley', 'Utilities', 'Heating', 'Street']

        for dataset in self.data:
            drop_columns(dataset, columns)
