set -e

TARGET_PATH='data/split/fold_target_outlier.csv'
STACKING_FILE='models/stacking/models_to_use.json'
NUM_FOLDS=5
ID_COLUMN='Id'
TARGET_COLUMN='SalePrice'

kaggleflow 'stacking' \
  --target-path=$TARGET_PATH \
  --stacking-file=$STACKING_FILE \
  --num-folds=$NUM_FOLDS \
  --id-column=$ID_COLUMN \
  --target-column=$TARGET_COLUMN
