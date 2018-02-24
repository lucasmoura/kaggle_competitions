import argparse


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

    print(user_args['train_file'])
    print(user_args['test_file'])


if __name__ == '__main__':
    main()
