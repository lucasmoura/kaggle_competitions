from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle


def split_data(dataset, test_size=0.2, seed=42):
    """
        Function used to load the original dataset and create the train
        and test split, used for evaluation purposes.

        :param dataset: A pandas DataFrame containing the original data.
        :param save_folder_name: Location to save the splitted DataFrames.
        :param train_name: Name of the train part of the split.
        :param test_name: Name of the test part of the split.
        :param test_size: Size of the test data to be created.
        :param seed: Seed used to control if the split will be deterministic.
    """

    train, test = train_test_split(dataset, test_size=test_size, random_state=seed)

    return train, test


def prepare_kfold(dataset, num_folds, seed):
    kf = KFold(num_folds, random_state=seed)
    kf.get_n_splits(dataset)

    return kf


def create_folds(dataset, num_folds, seed=42):
    """
        Function used to create the folds for training and evaluating our model.

        We will create an extra column on the dataset DataFrama called 'fold',
        which will indicate to which folder a data point belongs.

        :param dataset: A pandas DataFrame.
        :param num_folds: The number of folds to use.
        :param seed: Seed used to control if the folds will be deterministic.
    """

    dataset = dataset.assign(fold=None)
    dataset = shuffle(dataset, random_state=seed)

    kf = prepare_kfold(dataset, num_folds, seed)

    for fold_index, (_, test_indexes) in enumerate(kf.split(dataset)):
        dataset.iloc[test_indexes, dataset.columns.get_loc('fold')] = fold_index

    return dataset
