import pandas as pd
import numpy as np

from scipy.special import boxcox1p
from scipy.stats import skew

from sklearn.preprocessing import StandardScaler


class Dataset:

    def __init__(self, train_path, test_path, verbose=True):
        self.train_path = train_path
        self.test_path = test_path
        self.verbose = verbose

        self.categorical_cols = {}
        self.numeric_cols = {}
        self.bucket_cols = {}

    def load_datasets(self):
        self.train = pd.read_csv(self.train_path)
        self.test = pd.read_csv(self.test_path)

        if self.verbose:
            self.print_shape()

    def print_shape(self):
        print('Train shape ', self.train.shape)
        print('Test shape ', self.test.shape)

    def remove_above_ground_area_outliers(self):
        self.train = self.train.drop(
            self.train[
                (self.train['GrLivArea'] > 4000) &
                (self.train['SalePrice'] < 300000)].index
        )

    def fill_values_with_none(self, dataset):
        fill_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                     'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                     'MasVnrType', 'MSSubClass', 'BsmtQual', 'BsmtCond',
                     'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

        for column in fill_none:
            dataset[column] = dataset[column].fillna('None')

        return dataset

    def fill_values_with_zero(self, dataset):
        fill_zero = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                     'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea',
                     'GarageYrBlt', 'GarageArea', 'GarageCars']

        for column in fill_zero:
            dataset[column] = dataset[column].fillna(0)

        return dataset

    def fill_values_with_mode(self, dataset):
        dataset['Electrical'] = dataset['Electrical'].fillna(
            dataset['Electrical'].mode()[0])
        dataset['KitchenQual'] = dataset['KitchenQual'].fillna(
            dataset['KitchenQual'].mode()[0])
        dataset['Exterior1st'] = dataset['Exterior1st'].fillna(
            dataset['Exterior1st'].mode()[0])
        dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna(
            dataset['Exterior2nd'].mode()[0])
        dataset['SaleType'] = dataset['SaleType'].fillna(
            dataset['SaleType'].mode()[0])
        dataset['MSZoning'] = dataset['MSZoning'].fillna(
            dataset['MSZoning'].mode()[0])
        dataset["Functional"] = dataset["Functional"].fillna("Typ")

        return dataset

    def fill_values_with_median(self, dataset):
        dataset['LotFrontage'] = dataset.groupby(
            'Neighborhood')['LotFrontage'].transform(
                lambda x: x.fillna(x.median()))

        return dataset

    def column_values_to_str(self, dataset):
        dataset['MSSubClass'] = dataset['MSSubClass'].apply(str)
        dataset['OverallQual'] = dataset['OverallQual'].apply(str)
        dataset['OverallCond'] = dataset['OverallCond'].astype(str)
        dataset['YrSold'] = dataset['YrSold'].astype(str)
        dataset['MoSold'] = dataset['MoSold'].astype(str)

        return dataset

    def drop_values(self, dataset):
        dataset = dataset.drop(['Utilities'], axis=1)
        dataset = dataset.drop(['SalePrice'], axis=1)
        dataset = dataset.drop(['Id'], axis=1)

        return dataset

    def create_new_variable(self, dataset):
        dataset['TotalSF'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']  # noqa
        return dataset

    def create_categorical_columns(self, dataset):
        self.categorical_cols = {}

        for value in dataset.select_dtypes(include=[np.object]):
            self.categorical_cols[value] = dataset[value].unique().tolist()

    def create_numeric_columns(self, dataset):
        self.numeric_cols = dataset.columns.tolist() - self.categorical_cols.keys()  # noqa

    def create_dataset(self):
        self.load_datasets()
        self.remove_above_ground_area_outliers()
        self.test_id = self.test['Id']

        ntrain = self.train.shape[0]
        y_train = self.train.SalePrice.values

        dataset = pd.concat((self.train, self.test)).reset_index(drop=True)
        dataset = self.fill_values_with_none(dataset)
        dataset = self.fill_values_with_zero(dataset)
        dataset = self.fill_values_with_mode(dataset)
        dataset = self.fill_values_with_median(dataset)
        dataset = self.column_values_to_str(dataset)
        dataset = self.drop_values(dataset)
        dataset = self.create_new_variable(dataset)

        numeric_feats = dataset.dtypes[dataset.dtypes != "object"].index
        skewed_feats = dataset[numeric_feats].apply(
            lambda x: skew(x.dropna())).sort_values(ascending=False)
        skewness = pd.DataFrame({'Skew': skewed_feats})
        skewness = skewness[abs(skewness) > 0.5]
        skewed_features = skewness.index
        lam = 0.15
        for feat in skewed_features:
            dataset[feat] = boxcox1p(dataset[feat], lam)

        self.create_categorical_columns(dataset)
        self.create_numeric_columns(dataset)

        self.train = dataset[:ntrain]
        self.test = dataset[ntrain:]
        self.targets = pd.DataFrame(np.log(y_train))

        numerical_features = self.train.select_dtypes(
            exclude=["object"]).columns
        normalizer = StandardScaler()
        self.train.loc[:, numerical_features] = normalizer.fit_transform(
            self.train.loc[:, numerical_features])
        self.test.loc[:, numerical_features] = normalizer.transform(
            self.test.loc[:, numerical_features])

        return self.train, self.test, self.targets, self.test_id
