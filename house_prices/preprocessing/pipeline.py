class Pipeline:
    def __init__(self):
        self.train_data = None
        self.validation_data = None
        self.test_data = None

    def set_dataset(self, train, validation, test):
        self.data = (train, validation, test)

    def set_operations(self, fill_missing, transform, drop, create):
        self.operations = (fill_missing, transform, drop, create)

    def set_finalize(self, finalize):
        self.finalize = finalize

    def run_operations(self):
        train, validation, test = self.data

        for operation in self.operations:
            operation.set_dataset(train, validation, test)
            operation.run()
            train, validation, test = operation.get_dataset()

        return train, validation, test

    def run_pipeline(self):
        train, validation, test = self.run_operations()

        self.train_data = self.finalize.finalize_train(train)
        self.validation_data = self.finalize_validation(validation)
        self.test_data = self.finalize_test(test)

    @property
    def train_data(self):
        return self.train_data

    @property
    def validation_data(self):
        return self.validation_data

    @property
    def test_data(self):
        return self.test_data


class Operation:

    def __init__(self, signature):
        self.signature = signature

    def set_dataset(self, train, validation, test):
        self.data = (train, validation, test)

    def get_dataset(self):
        return self.data

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

    def __init__(self):
        super().__init__(signature='fill')


class Transformations(Operation):

    def __init__(self):
        super().__init__(signature='transform')


class Create(Operation):

    def __init__(self):
        super().__init__(signature='create')


class Drop(Operation):

    def __init__(self):
        super().__init__(signature='drop')


class Finalize:

    def __init__(self, train, validation, test):
        self.data = (train, validation, test)

    def finalize_train(self, train):
        raise NotImplementedError

    def finalize_validation(self, validation):
        raise NotImplementedError

    def finalize_test(self, test):
        raise NotImplementedError
