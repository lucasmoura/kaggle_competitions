from sklearn.ensemble import RandomForestRegressor

from kaggleflow.manager.base_model import Model


class RandomForest(Model):

    def create_model(self):
        self.random_forest = RandomForestRegressor()

    def fit(self, train_x, train_y):
        self.random_forest.fit(train_x, train_y)

    def set_config(self, config):
        self.random_forest.set_params(**config)

    def predict(self, test_x):
        return self.random_forest.predict(test_x)
