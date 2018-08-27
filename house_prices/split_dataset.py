import argparse

from preprocessing.load import load_dataset, save_dataset
from preprocessing.split import split_data, create_folds
from utils.folder import create_folder


def create_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dp',
                        '--dataset-path',
                        type=str,
                        help='Location of the original dataset')

    parser.add_argument('-nf',
                        '--num-folds',
                        type=int,
                        help='Number of folds to be created')

    parser.add_argument('-ts',
                        '--test-size',
                        type=float,
                        help='Percentage of the data used for the test split')

    parser.add_argument('-sv',
                        '--save-folder',
                        type=str,
                        help='Path to save the splitted dataset on')

    parser.add_argument('-tn',
                        '--train-name',
                        type=str,
                        help='Name of the train split when saved')

    parser.add_argument('-tsn',
                        '--test-name',
                        type=str,
                        help='Name of the test split when saved')

    return parser


def main():
    parser = create_argparse()
    user_args = vars(parser.parse_args())

    dataset_path = user_args['dataset_path']
    num_folds = user_args['num_folds']
    test_size = user_args['test_size']
    save_folder = user_args['save_folder']
    train_name = user_args['train_name']
    test_name = user_args['test_name']

    create_folder(save_folder)

    dataset = load_dataset(dataset_path, verbose=True)
    train_dataset, test_dataset = split_data(dataset, test_size=test_size)
    train_dataset = create_folds(train_dataset, num_folds=num_folds)

    save_dataset(train_dataset, save_folder, train_name)
    save_dataset(test_dataset, save_folder, test_name)


if __name__ == '__main__':
    main()
