import importlib
import inspect
import pkgutil


INVALID_PATH = -1


class ClassSearcher:

    def __init__(self, folder_to_look, is_pkg=True):
        self.folder_to_look = folder_to_look
        self.path = None
        self.is_pkg = is_pkg

    def iterate_over_module(self, path):
        if type(path) != list:
            path = [path]

        return pkgutil.iter_modules(path)

    def create_import_name(self, name, folder):
        return folder.replace('/', '.') + '.' + name

    def find_module(self, module_name):
        parsed_look_path = self.folder_to_look.replace('.', '/')

        for finder, name, ispkg in self.iterate_over_module(parsed_look_path):
            if name == module_name and self.is_pkg == ispkg:
                return self.create_import_name(name, self.folder_to_look)

        return None

    def get_class(self, name):
        self.path = self.find_module(name)

        if not self.path:
            return INVALID_PATH

        return self.load_module(self.path)

    def create_module_path(self, module_path):
        return self.create_import_name(self.default_module, module_path)

    def load_module_classes(self, module_path):
        module = importlib.import_module(module_path)
        return inspect.getmembers(module, inspect.isclass)

    def module_defined_classes(self, classes, module_path):
        return [class_obj for _, class_obj in classes
                if class_obj.__module__ == module_path]

    def load_module(self, module_path):
        module_path = self.create_module_path(module_path)
        model_classes = self.load_module_classes(module_path)
        return self.module_defined_classes(model_classes, module_path)


class ModelSearcher(ClassSearcher):
    def __init__(self, folder_to_look, is_pkg=True):
        super().__init__(folder_to_look, is_pkg)
        self.default_module = 'model'

    def load_module(self, module_path):
        module_defined_classes = super().load_module(module_path)
        return module_defined_classes[0]


class PipelineSearcher(ClassSearcher):
    def __init__(self, folder_to_look):
        super().__init__(folder_to_look)
        self.add_folder_to_search_at()
        self.default_module = 'pipeline'

    def add_folder_to_search_at(self):
        if not self.folder_to_look.endswith('pipelines'):
            self.folder_to_look = self.create_import_name(
                'pipelines', self.folder_to_look)

    def sort_operations(self, operations_classes):
        return sorted(operations_classes, key=lambda x: x.ORDER)

    def load_module(self, module_path):
        module_defined_classes = super().load_module(module_path)
        return self.sort_operations(module_defined_classes)


class MetricSearcher(ModelSearcher):
    def __init__(self, folder_to_look):
        super().__init__(folder_to_look, is_pkg=False)

    def create_module_path(self, module_path):
        return module_path
