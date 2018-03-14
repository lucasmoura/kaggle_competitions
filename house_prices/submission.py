import argparse

from preprocessing.dataset import Dataset


def create_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t',
                        '--train-file',
                        type=str,
                        help='Location of the train file (csv)')

    parser.add_argument('-ts',
                        '--test-file',
                        type=str,
                        help='Location of the test file (csv)')

    return parser


def main():
    parser = create_argparse()
    user_args = vars(parser.parse_args())

    train_file = user_args['train_file']
    test_file = user_args['test_file']

    print('Creating train dataset ...')
    train_dataset = Dataset(train_file)
    train_data = train_dataset.create_dataset()
    train_targets = train_dataset.targets

    print('Creating test dataset ...')
    test_data = Dataset(test_file, train=False).create_dataset()


if __name__ == '__main__':
    main()
