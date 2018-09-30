from xgboost import XGBRegressor

from kaggleflow.manager.base_model import Model


class XGBoostRegressor(Model):

    def create_model(self):
        self.xgb_regressor = XGBRegressor()

    def fit(self, train_x, train_y):
        self.xgb_regressor.fit(train_x, train_y)

    def set_config(self, config):
        self.xgb_regressor.set_params(**config)

    def predict(self, test_x):
        return self.xgb_regressor.predict(test_x)
