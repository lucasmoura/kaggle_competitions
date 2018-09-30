set -e

DATASET_PATH='data/outlier/train.csv'
NUM_FOLDS=5
SAVE_FOLDER='data/split/'
TRAIN_NAME='fold_train_outlier.csv'
TARGET_NAME='fold_target_outlier.csv'
TARGET_COLUMN='SalePrice'

kaggleflow 'split' \
  --dataset-path=$DATASET_PATH \
  --num-folds=$NUM_FOLDS \
  --save-folder=$SAVE_FOLDER \
  --train-name=$TRAIN_NAME \
  --target-name=$TARGET_NAME \
  --target-column=$TARGET_COLUMN
