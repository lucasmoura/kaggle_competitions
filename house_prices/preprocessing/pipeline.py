
class Pipeline:

    def __init__(self, train, validation, test, signature):
        self.data = (train, validation, test)
        self.signature = signature

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


class FillMissing(Pipeline):

    def __init__(self, train, validation, test):
        super().__init__(train, validation, test, signature='fill')


class Transformations(Pipeline):

    def __init__(self, train, validation, test):
        super().__init__(train, validation, test, signature='transform')


class Create(Pipeline):

    def __init__(self, train, validation, test):
        super().__init__(train, validation, test, signature='create')


class Drop(Pipeline):

    def __init__(self, train, validation, test):
        super().__init__(train, validation, test, signature='drop')
