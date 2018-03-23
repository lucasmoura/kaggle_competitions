import pandas as pd
import numpy as np


class Dataset:

    def __init__(self, dataset_path, train=True, verbose=True):
        self.dataset_path = dataset_path
        self.train = train
        self.verbose = verbose

    def load_dataset(self):
        self.dataset = pd.read_csv(self.dataset_path)

        if self.verbose:
            self.print_shape()

    def print_shape(self):
        print('Dataset shape ', self.dataset.shape)

    def encode_street(self):
        self.dataset['enc_street'] = pd.get_dummies(
            self.dataset.Street, drop_first=True)

    def encode_sales_condition(self):
        self.dataset['enc_condition'] = self.dataset.SaleCondition.apply(
            lambda x: 1 if x == 'Partial' else 0)

    def apply_feature_scaling(self, column_name):
        column_values = self.dataset[column_name]
        column_mean = self.dataset[column_name].mean()
        column_std = self.dataset[column_name].std()

        self.dataset[column_name] = (column_values - column_mean) / column_std

    def update_ms_sub_class(self):
        values = [20, 30, 40, 45, 50, 60, 70, 75,
                  80, 85, 90, 120, 150, 160, 180, 190]
        value_dict = dict(zip(values, range(len(values))))
        self.dataset['MSSubClass'] = self.dataset.MSSubClass.apply(
            lambda x: value_dict[x])

    def create_dataset(self):
        self.load_dataset()
        self.encode_street()
        self.encode_sales_condition()
        self.update_ms_sub_class()

        #self.apply_feature_scaling('LotFrontage')
        #self.apply_feature_scaling('LotArea')
        #self.apply_feature_scaling('MasVnrArea')
        #self.apply_feature_scaling('BsmtFinSF1')
        #self.apply_feature_scaling('BsmtFinSF2')
        #self.apply_feature_scaling('1stFlrSF')
        #self.apply_feature_scaling('2ndFlrSF')
        #self.apply_feature_scaling('GarageArea')
        #self.apply_feature_scaling('WoodDeckSF')
        #self.apply_feature_scaling('OpenPorchSF')
        #self.apply_feature_scaling('EnclosedPorch')
        #self.apply_feature_scaling('ScreenPorch')


class TrainDataset(Dataset):

    def remove_garage_area_outliers(self):
        self.dataset = self.dataset[self.dataset['GarageArea'] < 1200]

    def remove_null_values(self):
        self.dataset = self.dataset.select_dtypes(
            include=[np.number]).interpolate().dropna()

    def create_targets(self):
        self.targets = np.log(self.dataset.SalePrice)

    def drop_columns(self):
        self.dataset = self.dataset.drop(['SalePrice', 'Id'], axis=1)

    def create_dataset(self):
        super().create_dataset()

        self.remove_garage_area_outliers()
        self.remove_null_values()
        self.create_targets()
        self.drop_columns()

        if self.verbose:
            self.print_shape()

        return self.dataset


class TestDataset(Dataset):

    def select_features(self):
        ids = self.dataset['Id']

        self.dataset = self.dataset.select_dtypes(
            include=[np.number]).drop(['Id'], axis=1).interpolate()

        self.dataset['Id'] = ids

    def create_dataset(self):
        super().create_dataset()

        self.select_features()

        if self.verbose:
            self.print_shape()

        return self.dataset
