from preprocessing.load import load_dataset, save_dataset
from preprocessing.split import create_folds
from utils.folder import create_folder


def split_runner(args):
    dataset_path = args['dataset_path']
    num_folds = args['num_folds']
    save_folder = args['save_folder']
    train_name = args['train_name']

    create_folder(save_folder)

    print('Loading train dataset')
    train = load_dataset(dataset_path, verbose=True)

    print('Creating folds for cross-validation')
    train_folds = create_folds(train, num_folds=num_folds)

    print('Saving train fold dataset')
    save_dataset(train_folds, save_folder, train_name)


def create_split_parser(subparser):
    parse_split = subparser.add_parser('split')

    parse_split.add_argument(
        '-dp',
        '--dataset-path',
        type=str,
        help='Location of the original dataset')

    parse_split.add_argument(
        '-nf',
        '--num-folds',
        type=int,
        help='Number of folds to be created')

    parse_split.add_argument(
        '-sv',
        '--save-folder',
        type=str,
        help='Path to save the splitted dataset on')

    parse_split.add_argument(
        '-tn',
        '--train-name',
        type=str,
        help='Name of the train split when saved')

    parse_split.set_defaults(func=split_runner)
