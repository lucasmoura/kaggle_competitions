set -e

#TRAIN_PATH='data/split/fold_train.csv'
#TARGET_PATH='data/split/fold_target.csv'
TRAIN_PATH='data/split/fold_train_outlier.csv'
TARGET_PATH='data/split/fold_target_outlier.csv'
TEST_PATH='data/test.csv'
MODEL_NAME=$1
PIPELINE_NAME=$2
NUM_FOLDS=5
CREATE_SUBMISSION=1
USE_STACKING=1
ID_COLUMN='Id'
TARGET_COLUMN='SalePrice'

kaggleflow 'model' \
  --train-path=$TRAIN_PATH \
  --target-path=$TARGET_PATH \
  --test-path=$TEST_PATH \
  --model-name=$MODEL_NAME \
  --pipeline-name=$PIPELINE_NAME \
  --num-folds=$NUM_FOLDS \
  --create-submission=$CREATE_SUBMISSION \
  --use-stacking=$USE_STACKING \
  --id-column=$ID_COLUMN \
  --target-column=$TARGET_COLUMN
