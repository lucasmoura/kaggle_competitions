#!/bin/bash

set -e

source scripts/base_config.sh

python eda.py \
  --item-categories-path=$ITEM_CATEGORIES_PATH \
  --items-path=$ITEMS_PATH \
  --sales-train-path=$SALES_TRAIN_PATH \
  --shops-path=$SHOPS_PATH \
  --test-path=$TEST_PATH
