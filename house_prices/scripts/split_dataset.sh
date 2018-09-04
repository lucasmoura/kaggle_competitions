set -e

DATASET_PATH='data/train.csv'
NUM_FOLDS=5
SAVE_FOLDER='data/split/'
TRAIN_NAME='fold_train.csv'

python split_dataset.py \
  --dataset-path=$DATASET_PATH \
  --num-folds=$NUM_FOLDS \
  --save-folder=$SAVE_FOLDER \
  --train-name=$TRAIN_NAME \
