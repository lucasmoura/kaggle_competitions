from sklearn.linear_model import Ridge as RidgeLR

from manager.base_model import Model


class Ridge(Model):

    def create_model(self):
        self.ridge = RidgeLR()

    def fit(self, train_x, train_y):
        self.ridge.fit(train_x, train_y)

    def set_config(self, config):
        self.ridge.set_params(**config)

    def predict(self, test_x):
        return self.ridge.predict(test_x)
