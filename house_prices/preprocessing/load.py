import os

import pandas as pd


def print_shape(dataset):
    print('Dataset shape ', dataset.shape)


def load_dataset(dataset_path, verbose=False):
    dataset = pd.read_csv(dataset_path)

    if verbose:
        print_shape(dataset)

    return dataset


def save_dataset(dataset, save_folder, save_name):
    dataset.to_csv(os.path.join(save_folder, save_name), index=False)
