from models.base_pipeline import (BaseFillMissing, BaseTransformations,
                                  BaseCreate, BaseDrop, BaseFinalize)


class LFillMissing(BaseFillMissing):
    pass


class LTransformations(BaseTransformations):
    pass


class LCreate(BaseCreate):
    pass


class LDrop(BaseDrop):
    pass


class Finalize(BaseFinalize):
    pass
