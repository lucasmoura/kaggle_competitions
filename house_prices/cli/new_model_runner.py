from creator.model_creator import create_model


def new_model_runner(args):
    model_name = args['model_name']
    pipeline_name = args['pipeline_name']

    create_model('models', model_name, pipeline_name)


def create_new_model_parser(subparser, model_info_parser):

    new_model_parser = subparser.add_parser(
        'newmodel', parents=[model_info_parser])

    new_model_parser.set_defaults(func=new_model_runner)
