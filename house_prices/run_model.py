import argparse

from manager.model_manager import ModelEvaluation
from preprocessing.load import load_dataset


def create_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-tp',
                        '--train-path',
                        type=str,
                        help='Location of the training data')

    parser.add_argument('-tsp',
                        '--test-path',
                        type=str,
                        help='Location of the test data')

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

    return parser


def main():
    parser = create_argparse()
    user_args = vars(parser.parse_args())

    train = load_dataset(user_args['train_path'])
    test = load_dataset(user_args['test_path'])

    model_name = user_args['model_name']
    pipeline_name = user_args['pipeline_name']
    num_folds = user_args['num_folds']

    model_evaluation = ModelEvaluation(
        train, test, model_name, pipeline_name, num_folds)

    model_evaluation.run()


if __name__ == '__main__':
    main()
