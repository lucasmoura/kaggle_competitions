set -e

TRAIN_PATH='data/split/fold_train.csv'
MODEL_NAME=$1
PIPELINE_NAME=$2
NUM_FOLDS=5
NUM_ITER=30

python run_tuning.py \
  --train-path=$TRAIN_PATH \
  --model-name=$MODEL_NAME \
  --pipeline-name=$PIPELINE_NAME \
  --num-folds=$NUM_FOLDS \
  --num-iter=$NUM_ITER
