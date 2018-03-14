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

    def create_targets(self):
        self.targets = np.log(self.dataset.SalePrice)

    def remove_garage_area_outliers(self):
        self.dataset = self.dataset[self.dataset['GarageArea'] < 1200]

    def remove_null_values(self):
        self.dataset = self.dataset.select_dtypes(
            include=[np.number]).interpolate().dropna()

    def encode_street(self):
        self.dataset['enc_street'] = pd.get_dummies(
            self.dataset.Street, drop_first=True)

    def encode_sales_condition(self):
        self.dataset['enc_condition'] = self.dataset.SaleCondition.apply(
            lambda x: 1 if x == 'Partial' else 0)

    def drop_columns(self):
        self.dataset = self.dataset.drop(['SalePrice', 'Id'], axis=1)

    def select_features(self):
        self.dataset = self.dataset.select_dtypes(
            include=[np.number]).drop(['Id'], axis=1).interpolate()

    def create_dataset(self):
        self.load_dataset()
        self.encode_street()
        self.encode_sales_condition()

        if self.train:
            self.remove_garage_area_outliers()
            self.remove_null_values()
            self.create_targets()
            self.drop_columns()
        else:
            self.select_features()

        if self.verbose:
            self.print_shape()

        return self.dataset
