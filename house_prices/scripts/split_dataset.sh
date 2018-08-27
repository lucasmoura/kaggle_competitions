set -e

DATASET_PATH='data/train.csv'
NUM_FOLDS=5
TEST_SIZE=0.2
SAVE_FOLDER='data/split/'
TRAIN_NAME='fold_train.csv'
TEST_NAME='test.csv'

python split_dataset.py \
  --dataset-path=$DATASET_PATH \
  --num-folds=$NUM_FOLDS \
  --test-size=$TEST_SIZE \
  --save-folder=$SAVE_FOLDER \
  --train-name=$TRAIN_NAME \
  --test-name=$TEST_NAME
