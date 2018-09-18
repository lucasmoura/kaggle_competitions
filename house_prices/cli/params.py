import argparse

from cli.model_runner import create_model_parser
from cli.tuning_runner import create_tuning_parser
from cli.split_runner import create_split_parser


def create_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        '-tp',
        '--train-path',
        type=str,
        help='Location of the training data'
    )

    parser.add_argument(
        '-tgp',
        '--target-path',
        type=str,
        help='Location of the target path'
    )

    parser.add_argument(
        '-mn',
        '--model-name',
        type=str,
        help='Name of the model to be used'
    )

    parser.add_argument(
        '-pn',
        '--pipeline-name',
        type=str,
        help='Name of the pipeline to be used'
    )

    parser.add_argument(
        '-nf',
        '--num-folds',
        type=int,
        help='Number of folds to be used in cross validation'
    )

    return parser


def create_subparsers(parser):
    return parser.add_subparsers(title='actions')


def get_parser():
    parser = create_parser()
    subparser = create_subparsers(parser)

    create_tuning_parser(subparser, parser)
    create_model_parser(subparser, parser)
    create_split_parser(subparser)

    return parser
