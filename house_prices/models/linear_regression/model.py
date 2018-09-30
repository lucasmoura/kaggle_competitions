from sklearn.linear_model import LinearRegression as LR

from kaggleflow.manager.base_model import Model


class LinearRegression(Model):

    def create_model(self):
        self.linear_regression = LR()

    def fit(self, train_x, train_y):
        self.linear_regression.fit(train_x, train_y)

    def set_config(self, config):
        self.linear_regression.set_params(**config)

    def predict(self, test_x):
        return self.linear_regression.predict(test_x)
