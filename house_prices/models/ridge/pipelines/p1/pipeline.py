from models.base_pipeline import (BaseFillMissing, BaseTransformations,
                                  BaseCreate, BaseDrop, BaseTargetTransform,
                                  BaseFinalize, BasePredictionTransform)


class RFillMissing(BaseFillMissing):
    pass


class RTransformations(BaseTransformations):
    pass


class RCreate(BaseCreate):
    def __init__(self):
        super().__init__()
        self.category_columns.remove('GarageYrBlt')

    def create_age_at_selling_point(self):
        for dataset in self.loop_datasets():
            dataset['age_selling_point'] = dataset['YrSold'] - dataset['YearBuilt']

    def create_time_since_remodelled(self):
        for dataset in self.loop_datasets():
            dataset['time_since_remodelled'] = dataset['YrSold'] - dataset['YearRemodAdd']


class RDrop(BaseDrop):
    def __init__(self):
        super().__init__()

        self.drop_columns.extend(
            ['YearBuilt', 'YearRemodAdd', 'YrSold', 'GarageYrBlt']
        )


class RTargetTransform(BaseTargetTransform):
    pass


class RFinalize(BaseFinalize):
    pass


class RPredictionTransform(BasePredictionTransform):
    pass
