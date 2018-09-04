class Model:

    def create_model(self):
        raise NotImplementedError

    def fit(self, train_x, train_y):
        raise NotImplementedError

    def set_evaluation_metric(self, metric):
        self.metric = metric

    def evaluate(self, x, y):
        raise NotImplementedError

    def predict(sefl, x):
        raise NotImplementedError
