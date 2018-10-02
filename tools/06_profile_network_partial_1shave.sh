#! /bin/bash

MODEL_FILENAME=output/inception-v1.meta
NUM_SHAVES=-s=1
INPUT_NODE_FLAG=-in=InceptionV1/InceptionV1/Conv2d_2b_1x1/Relu
OUTPUT_NODE_FLAG=-on=InceptionV1/InceptionV1/MaxPool_3a_3x3/MaxPool

mvNCProfile ${MODEL_FILENAME} ${NUM_SHAVES} ${INPUT_NODE_FLAG} ${OUTPUT_NODE_FLAG}
