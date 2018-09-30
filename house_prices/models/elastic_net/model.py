from sklearn.linear_model import ElasticNet

from kaggleflow.manager.base_model import Model


class ElasticNetModel(Model):

    def create_model(self):
        self.elastic_net = ElasticNet()

    def fit(self, train_x, train_y):
        self.elastic_net.fit(train_x, train_y)

    def set_config(self, config):
        self.elastic_net.set_params(**config)

    def predict(self, test_x):
        return self.elastic_net.predict(test_x)
