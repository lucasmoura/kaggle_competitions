#!/bin/bash

#usage: ./scripts/run_submission.sh

set -e

COMPETITION_NAME='house-prices-advanced-regression-techniques'
DATA_FOLDER='data/'

TRAIN_FILE=$DATA_FOLDER$COMPETITION_NAME'/train.csv'
TEST_FILE=$DATA_FOLDER$COMPETITION_NAME'/test.csv'
NUM_EPOCHS=60
BATCH_SIZE=32
USE_VALIDATION=1

python submission.py \
  --train-file=${TRAIN_FILE} \
  --test-file=${TEST_FILE} \
  --num-epochs=${NUM_EPOCHS} \
  --batch-size=${BATCH_SIZE} \
  --use-validation=${USE_VALIDATION}
