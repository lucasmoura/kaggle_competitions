import tensorflow as tf


class LinearRegressionEstimator:

    def get_numeric_columns(self):
        lot_area = tf.feature_column.numeric_column(
            key='LotArea'
        )

        lot_frontage = tf.feature_column.numeric_column(
            key='LotArea'
        )

        mas_vnr_area = tf.feature_column.numeric_column(
            key='ManVnrArea'
        )

        bsmt_fin_sf1 = tf.feature_column.numeric_column(
            key='BsmtFinSF1'
        )

        bsmt_fin_sf2 = tf.feature_column.numeric_column(
            key='BsmtFinSF2'
        )

        bsmt_unf_sf = tf.feature_column.numeric_column(
            key='BsmtUnfSF'
        )

        total_bsmt_sf = tf.feature_column.numeric_column(
            key='TotalBsmtSF'
        )

        1st_flr_sf = tf.feature_column.numeric_column(
            key='1stFlrSF'
        )

        2nd_flr_sf = tf.feature_column.numeric_column(
            key='2ndFlrSF'
        )

        garage_area = tf.feature_column.numeric_column(
            key='GarageArea'
        )

        wood_deck_sf = tf.feature_column.numeric_column(
            key='WoodDeckSF'
        )

    def get_bucketized_columns(self):
        year_build = tf.feature_column.numeric_column(
            key='YearBuilt'
        )
        year_built_bucket = tf.feature_column.bucketized_column(
            year_build,
            boundaries=[1900, 1950, 2000]
        )

        year_remod_add = tf.feature_column.numeric_column(
            key='YearRemodAdd'
        )
        year_built_bucket = tf.feature_column.bucketized_column(
            year_remod_add,
            boundaries=[1980, 2000]
        )

        gr_liv_area = tf.feature_column.numeric_column(
            key='GrLivArea'
        )
        gr_liv_are_bucket = tf.feature_column.bucketized_column(
            gr_liv_area,
            boundaries=[2200]
        )

        kitchen_abv_gt = tf.feature_column.numeric_column(
            key='KitchenAbvGr'
        )
        kitchen_abv_gt_bucket = tf.feature_column.bucketized_column(
            kitchen_abv_gt,
            boundaries=[1]
        )

        garage_yr_blt = tf.feature_column.numeric_column(
            key='GarageYrBlt'
        )
        garage_yr_blt_bucket = tf.feature_column.bucketized_column(
            garage_yr_blt,
            boundaries=[1900, 1960, 1980, 2000]
        )

    def get_feature_columns(self):
        ms_sub_class = tf.feature_column.categorical_column_with_identity(
            key='MSSubClass',
            num_buckets=17
        )

        overral_qual = tf.feature_column.categorical_column_with_identity(
            key='OverallQual',
            num_buckets=11
        )

        overral_cond = tf.feature_column.categorical_column_with_identity(
            key='OverallCond',
            num_buckets=11
        )

        bsmt_full_bath = tf.feature_column.categorical_column_with_identity(
            key='BsmtFullBath',
            num_buckets=4
        )

        bsmt_half_bath = tf.feature_column.categorical_column_with_identity(
            key='BsmtHalfBath',
            num_buckets=3
        )

        full_bath = tf.feature_column.categorical_column_with_identity(
            key='FullBath',
            num_buckets=4
        )

        half_bath = tf.feature_column.categorical_column_with_identity(
            key='HalfBath',
            num_buckets=3
        )

        bedroom_abv_gr = tf.feature_column.categorical_column_with_identity(
            key='BedroomAbvGr',
            num_buckets=9
        )

        tot_rms_abv_grd = tf.feature_column.categorical_column_with_identity(
            key='TotRmsAbvGrd',
            num_buckets=15
        )

        fireplaces = tf.feature_column.categorical_column_with_identity(
            key='Fireplaces',
            num_buckets=4
        )

        garage_cars = tf.feature_column.categorical_column_with_identity(
            key='GarageCars',
            num_buckets=5
        )

    def run_estimator(self):
        columns = self.get_feature_columns()

        model_dir = tempfile.mkdtemp()
        self.estimator = tf.estimator.LinearClassifier(
            model_dir=model_dir, feature_columns=columns)
