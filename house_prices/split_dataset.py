import argparse

from preprocessing.load import load_dataset, save_dataset
from preprocessing.split import create_folds
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

    parser.add_argument('-sv',
                        '--save-folder',
                        type=str,
                        help='Path to save the splitted dataset on')

    parser.add_argument('-tn',
                        '--train-name',
                        type=str,
                        help='Name of the train split when saved')

    return parser


def main():
    parser = create_argparse()
    user_args = vars(parser.parse_args())

    dataset_path = user_args['dataset_path']
    num_folds = user_args['num_folds']
    save_folder = user_args['save_folder']
    train_name = user_args['train_name']

    create_folder(save_folder)

    print('Loading train dataset')
    train = load_dataset(dataset_path, verbose=True)

    print('Creating folds for cross-validation')
    train_folds = create_folds(train, num_folds=num_folds)

    print('Saving train fold dataset')
    save_dataset(train_folds, save_folder, train_name)


if __name__ == '__main__':
    main()
