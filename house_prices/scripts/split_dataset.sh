set -e

DATASET_PATH='data/train.csv'
NUM_FOLDS=5
SAVE_FOLDER='data/split/'
TRAIN_NAME='fold_train.csv'
TARGET_NAME='fold_target.csv'
TARGET_COLUMN='SalePrice'

python run.py 'split' \
  --dataset-path=$DATASET_PATH \
  --num-folds=$NUM_FOLDS \
  --save-folder=$SAVE_FOLDER \
  --train-name=$TRAIN_NAME \
  --target-name=$TARGET_NAME \
  --target-column=$TARGET_COLUMN
