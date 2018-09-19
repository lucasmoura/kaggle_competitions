import argparse

from cli.model_runner import create_model_parser
from cli.tuning_runner import create_tuning_parser
from cli.split_runner import create_split_parser
from cli.stacking_runner import create_stacking_runner


def create_base_parser():
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
        '-nf',
        '--num-folds',
        type=int,
        help='Number of folds to be used in cross validation'
    )

    return parser


def create_model_info_parser():
    parser = argparse.ArgumentParser(add_help=False)

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

    return parser


def create_submission_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        '-ic',
        '--id-column',
        type=str,
        help='Name of the Id column')

    parser.add_argument(
        '-tc',
        '--target-column',
        type=str,
        help='Name of the target column')

    return parser


def create_subparsers(parser):
    return parser.add_subparsers(title='actions')


def get_parser():
    base_parser = create_base_parser()
    model_info_parser = create_model_info_parser()
    submission_parser = create_submission_parser()

    subparser = create_subparsers(base_parser)

    create_tuning_parser(subparser, base_parser, model_info_parser)
    create_model_parser(
        subparser, base_parser, model_info_parser, submission_parser)
    create_split_parser(subparser)
    create_stacking_runner(subparser, base_parser, submission_parser)

    return base_parser
