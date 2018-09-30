from kaggleflow.manager.model_manager import ModelEvaluation
from kaggleflow.manager.stacking.stacking_manager import StackingModel
from kaggleflow.preprocessing.load import load_dataset


def model_runner(args):
    train = load_dataset(args['train_path'])
    target = load_dataset(args['target_path'])
    test = load_dataset(args['test_path'])

    model_name = args['model_name']
    pipeline_name = args['pipeline_name']
    num_folds = args['num_folds']
    create_submission = args['create_submission']
    use_stacking = args['use_stacking']

    id_column = args['id_column']
    target_column = args['target_column']

    model = StackingModel if use_stacking else ModelEvaluation

    model_evaluation = model(
        train, target, test, model_name,
        pipeline_name, num_folds, create_submission,
        id_column, target_column
    )

    model_evaluation.run()


def create_model_parser(subparser, base_parser,
                        model_info_parser, submission_parser):
    parse_model = subparser.add_parser(
        'model', parents=[base_parser, model_info_parser, submission_parser]
    )

    parse_model.add_argument(
        '-tsp',
        '--test-path',
        type=str,
        help='Location of the test data')

    parse_model.add_argument(
        '-cs',
        '--create-submission',
        type=int,
        help='If a submission should be created for this model')

    parse_model.add_argument(
        '-us',
        '--use-stacking',
        type=int,
        help='If we should create a prediction file for stacking')

    parse_model.set_defaults(func=model_runner)
