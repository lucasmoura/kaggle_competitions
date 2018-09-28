from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from manager.base_model import Model


class LassoModel(Model):

    def create_model(self):
        self.lasso = Lasso()

    def fit(self, train_x, train_y):
        self.lasso_pip = make_pipeline(RobustScaler(), self.lasso)
        self.lasso_pip.fit(train_x, train_y)

    def set_config(self, config):
        self.lasso.set_params(**config)

    def predict(self, test_x):
        return self.lasso_pip.predict(test_x)
