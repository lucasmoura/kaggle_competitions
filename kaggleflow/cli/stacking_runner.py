from kaggleflow.preprocessing.load import load_dataset
from kaggleflow.manager.stacking.stacking_manager import StackingEvaluation


def stacking_runner(args):
    target = load_dataset(args['target_path'])

    stacking_file = args['stacking_file']
    num_folds = args['num_folds']
    id_column = args['id_column']
    target_column = args['target_column']

    stacking_runner = StackingEvaluation(
        target, stacking_file, num_folds, id_column,
        target_column)

    stacking_runner.run()


def create_stacking_runner(subparser, base_parser, submission_parser):
    parse_stacking = subparser.add_parser(
        'stacking', parents=[base_parser, submission_parser]
    )

    parse_stacking.add_argument(
        '-sf',
        '--stacking-file',
        type=str,
        help='Location of the stacking file')

    parse_stacking.set_defaults(func=stacking_runner)
