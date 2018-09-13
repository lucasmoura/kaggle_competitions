import argparse

from manager.tuning.tuning_manager import ModelTuning
from preprocessing.load import load_dataset


def create_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-tp',
                        '--train-path',
                        type=str,
                        help='Location of the training data')

    parser.add_argument('-mn',
                        '--model-name',
                        type=str,
                        help='Name of the model to be used')

    parser.add_argument('-pn',
                        '--pipeline-name',
                        type=str,
                        help='Name of the pipeline to be used')

    parser.add_argument('-nf',
                        '--num-folds',
                        type=int,
                        help='Number of folds to be used in cross validation')

    parser.add_argument('-ni',
                        '--num-iter',
                        type=int,
                        help='Number of iterations to perform model tuning')

    return parser


def main():
    parser = create_argparse()
    user_args = vars(parser.parse_args())

    train = load_dataset(user_args['train_path'])

    model_name = user_args['model_name']
    pipeline_name = user_args['pipeline_name']
    num_folds = user_args['num_folds']
    num_iter = user_args['num_iter']

    model_tuning = ModelTuning(
        train, model_name, pipeline_name, num_folds, num_iter)

    model_tuning.run()


if __name__ == '__main__':
    main()
