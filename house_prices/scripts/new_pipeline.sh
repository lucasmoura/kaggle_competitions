set -e

MODEL_NAME=$1
PIPELINE_NAME=$2

python run.py 'newpipeline' \
  --model-name=$MODEL_NAME \
  --pipeline-name=$PIPELINE_NAME \
