from kaggleflow.manager.tuning.tuning_manager import ModelTuning
from kaggleflow.preprocessing.load import load_dataset


def tuning_runner(args):
    train = load_dataset(args['train_path'])
    target = load_dataset(args['target_path'])

    model_name = args['model_name']
    pipeline_name = args['pipeline_name']
    num_folds = args['num_folds']
    num_iter = args['num_iter']

    model_tuning = ModelTuning(
        train, target, model_name, pipeline_name, num_folds, num_iter)

    model_tuning.run()


def create_tuning_parser(subparser, base_parser, model_info_parser):
    parse_tuning = subparser.add_parser(
        'tuning', parents=[base_parser, model_info_parser])

    parse_tuning.add_argument(
        '-ni',
        '--num-iter',
        type=int,
        help='Number of iterations to perform model tuning')

    parse_tuning.set_defaults(func=tuning_runner)
