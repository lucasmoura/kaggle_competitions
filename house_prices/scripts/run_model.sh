set -e

TRAIN_PATH='data/split/fold_train.csv'
TEST_PATH='data/test.csv'
MODEL_NAME='linear_regression'
PIPELINE_NAME='p1'
NUM_FOLDS=5
CREATE_SUBMISSION=1

python run_model.py \
  --train-path=$TRAIN_PATH \
  --test-path=$TEST_PATH \
  --model-name=$MODEL_NAME \
  --pipeline-name=$PIPELINE_NAME \
  --num-folds=$NUM_FOLDS \
  --create-submission=$CREATE_SUBMISSION
