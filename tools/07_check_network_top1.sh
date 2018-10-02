#! /bin/bash

MODEL_FILENAME=output/inception-v1.meta
INPUT_NODE_FLAG=-in=input
OUTPUT_NODE_FLAG=-on=InceptionV1/Logits/Predictions/Reshape_1

mvNCCheck -s 12 ${MODEL_FILENAME} ${INPUT_NODE_FLAG} ${OUTPUT_NODE_FLAG} -i images/cat.jpg -id 829 -S 2 -M 128 -cs 0,1,2 -metric top1
