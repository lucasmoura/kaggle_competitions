set -e

MODEL_NAME=$1
PIPELINE_NAME=$2

kaggleflow 'newpipeline' \
  --model-name=$MODEL_NAME \
  --pipeline-name=$PIPELINE_NAME \
