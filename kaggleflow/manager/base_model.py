class Model:

    def create_model(self):
        raise NotImplementedError

    def fit(self, train_x, train_y):
        raise NotImplementedError

    def set_evaluation_metric(self, metric):
        self.metric = metric

    def set_config(self, config):
        raise NotImplementedError

    def evaluate(self, predictions, target):
        return self.metric.compute(predictions, target)

    def predict(sefl, x):
        raise NotImplementedError
