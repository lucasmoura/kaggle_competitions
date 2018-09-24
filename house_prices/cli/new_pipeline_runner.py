import os

from creator.model_creator import create_pipeline


def new_pipeline_runner(args):
    model_name = args['model_name']
    pipeline_name = args['pipeline_name']

    model_path = os.path.join('models', model_name)
    create_pipeline(model_name, pipeline_name, model_path)


def create_new_pipeline_parser(subparser, model_info_parser):

    new_pipeline_parser = subparser.add_parser(
        'newpipeline', parents=[model_info_parser])

    new_pipeline_parser.set_defaults(func=new_pipeline_runner)
