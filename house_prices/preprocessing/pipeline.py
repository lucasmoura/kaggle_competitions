class Pipeline:
    def __init__(self):
        self.train_data = None
        self.validation_data = None
        self.test_data = None

    def set_dataset(self, train, validation, test):
        self.data = (train, validation, test)

    def set_operations(self, fill_missing, transform, create, drop):
        self.operations = (
            fill_missing(),
            transform(),
            create(),
            drop()
        )

    def set_finalize(self, finalize):
        self.finalize = finalize()

    def run_operations(self, verbose):
        train, validation, test = self.data

        for operation in self.operations:
            operation.set_dataset(train, validation, test)

            if verbose:
                print('Running pipeline operation {}'.format(
                    operation.name))

            operation.run()
            train, validation, test = operation.get_dataset()

        return train, validation, test

    def run_pipeline(self, verbose=True):
        train, validation, test = self.run_operations(verbose)

        self.train_data = self.finalize.finalize_train(train)

        if validation is not None:
            self.validation_data = self.finalize.finalize_validation(validation)

        self.test_data = self.finalize.finalize_test(test)


class Operation:

    ORDER = 0

    def __init__(self, signature):
        self.signature = signature

    def set_dataset(self, train, validation, test):
        self.data = (train, validation, test)

    def get_dataset(self):
        return self.data

    def loop_datasets(self):
        for dataset in self.data:
            if dataset is not None:
                yield dataset

    def get_signature_methods(self):
        methods = []

        for method in dir(self):
            if not method.startswith(self.signature):
                continue

            if callable(getattr(self, method)):
                methods.append(method)

        return methods

    def run_methods(self, methods):
        for method in methods:
            getattr(self, method)()

    def run(self):
        methods = self.get_signature_methods()
        self.run_methods(methods)


class FillMissing(Operation):

    ORDER = 1

    def __init__(self):
        super().__init__(signature='fill')
        self.name = 'Fill Missing'


class Transformations(Operation):

    ORDER = 2

    def __init__(self):
        super().__init__(signature='transform')
        self.name = 'Transformation'


class Create(Operation):

    ORDER = 3

    def __init__(self):
        super().__init__(signature='create')
        self.name = 'Create'


class Drop(Operation):

    ORDER = 4

    def __init__(self):
        super().__init__(signature='drop')
        self.name = 'Drop'


class Finalize:

    ORDER = 5

    def __init__(self):
        self.name = 'Finalize'

    def finalize_train(self, train):
        raise NotImplementedError

    def finalize_validation(self, validation):
        raise NotImplementedError

    def finalize_test(self, test):
        raise NotImplementedError


class PredictionTransform:

    ORDER = 6

    def transform_predictions(predictions):
        return predictions

    def revert_transform_predictions(predictions):
        return predictions
