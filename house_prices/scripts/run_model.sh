set -e

TRAIN_PATH='data/split/fold_train.csv'
TEST_PATH='data/test.csv'
MODEL_NAME=$1
PIPELINE_NAME=$2
NUM_FOLDS=5
CREATE_SUBMISSION=1
USE_STACKING=1

python run.py 'model' \
  --train-path=$TRAIN_PATH \
  --test-path=$TEST_PATH \
  --model-name=$MODEL_NAME \
  --pipeline-name=$PIPELINE_NAME \
  --num-folds=$NUM_FOLDS \
  --create-submission=$CREATE_SUBMISSION \
  --use-stacking=$USE_STACKING
