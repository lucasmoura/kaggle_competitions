from manager.model_manager import ModelEvaluation
from preprocessing.load import load_dataset


def model_runner(args):
    train = load_dataset(args['train_path'])
    test = load_dataset(args['test_path'])

    model_name = args['model_name']
    pipeline_name = args['pipeline_name']
    num_folds = args['num_folds']
    create_submission = args['create_submission']

    model_evaluation = ModelEvaluation(
        train, test, model_name, pipeline_name, num_folds, create_submission)

    model_evaluation.run()


def create_model_parser(subparser, parent):
    parse_model = subparser.add_parser('model', parents=[parent])

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

    parse_model.set_defaults(func=model_runner)
