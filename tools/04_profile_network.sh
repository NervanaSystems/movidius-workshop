#! /bin/bash

MODEL_FILENAME=output/inception-v1.meta
NUM_SHAVES=-s=12
INPUT_NODE_FLAG=-in=input
OUTPUT_NODE_FLAG=-on=InceptionV1/Logits/Predictions/Reshape_1

mvNCProfile ${MODEL_FILENAME} ${NUM_SHAVES} ${INPUT_NODE_FLAG} ${OUTPUT_NODE_FLAG}
