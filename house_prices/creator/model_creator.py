import os

from utils.folder import create_folder


def create_folders(model_path, pipelines_path, pipeline_name_path):
    create_folder(model_path)
    create_folder(pipelines_path)
    create_folder(pipeline_name_path)


def create_file(path, file_name):
    open(os.path.join(path, file_name), 'w').close()


def create_init_files(model_path, pipeline_name_path):
    file_name = '__init__.py'

    create_file(model_path, file_name)
    create_file(model_path, file_name)


def create_model_class_name(model_name):
    return model_name.title().replace('_', '')


def create_model_file(model_path, model_name):
    model_name = create_model_class_name(model_name)

    model_file = """
from manager.base_model import Model

class {}(Model):

    def create_model(self):
        pass

    def fit(self, train_x, train_y):
        pass

    def set_config(self, config):
        pass

    def predict(self, test_x):
        pass
    """

    model_file_path = os.path.join(model_path, 'model.py')

    with open(model_file_path, 'w') as f:
        f.write(model_file.format(model_name))


def create_pipeline_class_name(model_name):
    return ''.join(list(filter(str.isupper, model_name.title())))


def create_pipeline_file(pipeline_name_path, model_name):
    pipeline_base_name = create_pipeline_class_name(model_name)

    pipeline_file = """
from models.base_pipeline import (BaseFillMissing, BaseTransformations,
                                  BaseCreate, BaseDrop, BaseTargetTransform,
                                  BaseFinalize, BasePredictionTransform)


class {0}FillMissing(BaseFillMissing):
    pass


class {0}Transformations(BaseTransformations):
    pass


class {0}Create(BaseCreate):
    pass


class {0}Drop(BaseDrop):
    pass


class {0}TargetTransform(BaseTargetTransform):
    pass


class {0}Finalize(BaseFinalize):
    pass


class {0}PredictionTransform(BasePredictionTransform):
    pass
"""

    pipeline_name_file_path = os.path.join(pipeline_name_path, 'pipeline.py')
    with open(pipeline_name_file_path, 'w') as f:
        f.write(pipeline_file.format(pipeline_base_name))


def create_tuning_file(pipeline_name_path):
    create_file(pipeline_name_path, 'tuning.json')


def create_config_file(pipeline_name_path):
    create_file(pipeline_name_path, 'config.json')


def create_model(folder, model_name, pipeline_name):
    model_path = os.path.join(folder, model_name)
    pipelines_path = os.path.join(model_path, 'pipelines')
    pipeline_name_path = os.path.join(pipelines_path, pipeline_name)

    create_folders(model_path, pipelines_path, pipeline_name_path)
    create_init_files(model_path, pipeline_name_path)
    create_model_file(model_path, model_name)
    create_pipeline_file(pipeline_name_path, model_name)

    create_tuning_file(pipeline_name_path)
    create_config_file(pipeline_name_path)
