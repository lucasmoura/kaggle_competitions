import importlib
import inspect
import pkgutil


INVALID_MODEL_NAME = -1
INVALID_PIPELINE_NAME = -2


def iterate_over_module(path):
    if type(path) != list:
        path = [path]

    return pkgutil.iter_modules(path)


def create_import_name(model_path, name):
    return model_path.replace('/', '.') + '.' + name


def find_model(model_name, module='models'):
    for finder, name, ispkg in iterate_over_module(module):
        if name == model_name and ispkg:
            return create_import_name(module, name)

    return None


def find_pipeline(pipeline_name, model_path):
    pipeline_path = create_import_name(model_path, 'pipelines')
    pipeline_file_path = pipeline_path.replace('.', '/')

    for finder, name, ispkg in iterate_over_module(pipeline_file_path):
        if name == pipeline_name and ispkg:
            return create_import_name(pipeline_path, name)

    return None


def load_module_classes(path):
    module = importlib.import_module(path)
    return inspect.getmembers(module, inspect.isclass)


def load_model(model_path, module_name='model'):
    model_path = create_import_name(model_path, module_name)
    model_class = load_module_classes(model_path)[0][1]

    return model_class


def module_defined_classes(classes, module_name):
    return [class_obj for _, class_obj in classes
            if class_obj.__module__ == module_name]


def sort_operations(operations_classes):
    return sorted(operations_classes, key=lambda x: x.ORDER)


def load_pipeline_operations(pipeline_path, module_name='pipeline'):
    pipeline_path = create_import_name(pipeline_path, module_name)

    operations_classes = load_module_classes(pipeline_path)
    operations_defined_classes = module_defined_classes(operations_classes, pipeline_path)

    return sort_operations(operations_defined_classes)


def get_model_objects(model_name, pipeline_name):
    model_path = find_model(model_name)
    if not model_path:
        return INVALID_MODEL_NAME

    model = load_model(model_path)

    pipeline_path = find_pipeline(pipeline_name, model_path)
    if not pipeline_path:
        return INVALID_PIPELINE_NAME

    pipeline_operations = load_pipeline_operations(pipeline_path)

    return (model, pipeline_operations)
