from sklearn.linear_model import LinearRegression as LR
from manager.base_model import Model


class LinearRegression(Model):

    def create_model(self):
        self.linear_regression = LR()

    def fit(self, train_x, train_y):
        self.linear_regression.fit(train_x, train_y)

    def evaluate(self, validation_x, validation_y):
        predictions = self.linear_regression.predict(validation_x)
        return self.metric.compute(predictions, validation_y)

    def predict(self, test_x):
        return self.linear_regression.predict(test_x)
