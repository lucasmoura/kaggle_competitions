set -e

TRAIN_PATH='data/split/fold_train_outlier.csv'
TARGET_PATH='data/split/fold_target_outlier.csv'
MODEL_NAME=$1
PIPELINE_NAME=$2
NUM_FOLDS=5
NUM_ITER=$3

kaggleflow 'tuning' \
  --train-path=$TRAIN_PATH \
  --target-path=$TARGET_PATH \
  --model-name=$MODEL_NAME \
  --pipeline-name=$PIPELINE_NAME \
  --num-folds=$NUM_FOLDS \
  --num-iter=$NUM_ITER
