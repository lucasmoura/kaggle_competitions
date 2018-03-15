#!/bin/bash

#usage: ./scripts/run_submission.sh

set -e

COMPETITION_NAME='house-prices-advanced-regression-techniques'
DATA_FOLDER='data/'

TRAIN_FILE=$DATA_FOLDER$COMPETITION_NAME'/train.csv'
TEST_FILE=$DATA_FOLDER$COMPETITION_NAME'/test.csv'

python submission.py \
  --train-file=${TRAIN_FILE} \
  --test-file=${TEST_FILE}