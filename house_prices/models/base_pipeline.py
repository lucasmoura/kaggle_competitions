import numpy as np

from preprocessing.create import create_column
from preprocessing.pipeline import (Transformations, FillMissing,
                                    Create, Drop, TargetTransform,
                                    Finalize, PredictionTransform)
from preprocessing.transform import (transform_categorical_column,
                                     transform_column_into_categorical_dtype,
                                     transform_categorical_to_one_hot,
                                     transform_to_log1_scale)
from preprocessing.missing_data import fill_nan_with_value, drop_columns


class BaseFillMissing(FillMissing):

    def fill_na_categorical_columns(self):
        columns = ['BsmtQual', 'BsmtCond', 'FireplaceQu', 'GarageQual',
                   'GarageCond', 'GarageFinish', 'GarageType', 'GarageYrBlt',
                   'MasVnrType', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                   'Fence']

        for dataset in self.loop_datasets():
            fill_nan_with_value(dataset, columns, 'NP')

    def fill_mas_vnr_type(self):
        for dataset in self.loop_datasets():
            fill_nan_with_value(dataset, ['MasVnrType'], 'NP')

    def fill_with_zero(self):
        columns = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                   'TotalBsmtSF', 'GarageCars', 'GarageArea',
                   'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']

        for column in columns:
            for dataset in self.loop_datasets():
                fill_nan_with_value(dataset, [column], 0)

    def fill_lot_frontage(self):
        train = self.data[0]
        mean = train['LotFrontage'].mean()

        for dataset in self.loop_datasets():
            fill_nan_with_value(dataset, ['LotFrontage'], mean)

    def mode_fill(self, column):
        train = self.data[0]
        mode = train[column].mode()[0]

        for dataset in self.loop_datasets():
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

    def __init__(self):
        super().__init__()

        self.category_columns = [
            'MSSubClass', 'MSZoning', 'LotShape', 'LandContour', 'LotConfig',
            'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
            'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
            'MasVnrType', 'Foundation', 'CentralAir', 'Electrical',
            'Functional', 'GarageType', 'GarageYrBlt',
            'GarageFinish', 'PavedDrive',
            'SaleType', 'SaleCondition'
        ]

        self.numeric_columns = [
            'LotFrontage', 'LotArea', 'MasVnrArea',
            'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
            '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
            'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
            '3SsnPorch', 'ScreenPorch'
        ]

    def apply_ordinal_transformation(self, ordinal_map, columns):
        for column in columns:
            for dataset in self.loop_datasets():
                transform_categorical_column(dataset, column, ordinal_map)

    def transform_to_ordinal(self):
        ordinal_map = {'NP': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        columns = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'ExterQual', 'ExterCond',
                   'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond']
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
        for column in self.category_columns:
            for dataset in self.loop_datasets():
                transform_column_into_categorical_dtype(dataset, column)

    def transform_numeric_data(self):
        for column in self.numeric_columns:
            for dataset in self.loop_datasets():
                transform_to_log1_scale(dataset, column)


class BaseDrop(Drop):

    def drop_columns_from_datasets(self):
        columns = ['PoolQC', 'MiscFeature', 'Alley',
                   'Utilities', 'Heating', 'Street', 'Id', 'PoolArea']

        for dataset in self.loop_datasets():
            drop_columns(dataset, columns)


class BaseCreate(Create):
    def __init__(self):
        super().__init__()

        self.category_columns = [
            'MSSubClass', 'MSZoning', 'LotShape', 'LandContour', 'LotConfig',
            'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
            'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
            'MasVnrType', 'Foundation', 'CentralAir', 'Electrical',
            'Functional', 'GarageType', 'GarageYrBlt',
            'GarageFinish', 'PavedDrive', 'SaleType', 'SaleCondition'
        ]

    def handle_missing_columns(self):
        train, validation, test = self.data

        if validation is not None:
            self.handle_new_columns(train, validation)
            self.remove_additional_columns(validation, train)

        if test is not None:
            self.handle_new_columns(train, test)
            self.remove_additional_columns(test, train)

    def select_converted_categorical_columns(self, dataset):
        return list(dataset.select_dtypes(include=[np.uint8]).columns)

    def get_missing_columns(self, d1, d2):
        d1_valid_columns = self.select_converted_categorical_columns(d1)
        d2_valid_columns = self.select_converted_categorical_columns(d2)

        return set(d1_valid_columns) - set(d2_valid_columns)

    def handle_new_columns(self, d1, d2):
        missing_columns = self.get_missing_columns(d1, d2)

        for column in missing_columns:
            create_column(d2, column, 0)

    def remove_additional_columns(self, d1, d2):
        missing_columns = self.get_missing_columns(d1, d2)
        drop_columns(d1, missing_columns)

    def create_one_hot(self):
        """
        This solution is definetely not optimal, but since we need
        to perform the operations in place, this was the idea that I had.
        """

        for dataset in self.loop_datasets():
            new_dataset = transform_categorical_to_one_hot(
                dataset, self.category_columns)

            drop_columns(dataset, self.category_columns)
            dataset[new_dataset.columns] = new_dataset

        self.handle_missing_columns()


class BaseTargetTransform(TargetTransform):

    def transform_target(dataset):
        return np.log(dataset)


class BaseFinalize(Finalize):

    def create_supervised_dataset(self, dataset):
        X = dataset[self.columns_order]

        return X

    def set_column_order(self, train):
        train_columns = train.columns.tolist()
        train_columns.remove('fold')

        self.columns_order = train_columns

    def finalize_train(self, train):
        self.set_column_order(train)
        return self.create_supervised_dataset(train)

    def finalize_validation(self, validation):
        return self.create_supervised_dataset(validation)

    def finalize_test(self, test):
        return test[self.columns_order]


class BasePredictionTransform(PredictionTransform):

    def transform_predictions(predictions):
        return predictions

    def revert_transform_predictions(predictions):
        return np.exp(predictions)
