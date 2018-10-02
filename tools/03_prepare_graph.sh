#! /bin/bash

MODEL_FILENAME=output/inception-v1.meta
INPUT_NODE_FLAG=-in=input
OUTPUT_NODE_FLAG=-on=InceptionV1/Logits/Predictions/Reshape_1
OUTPUT_GRAPH_FILENAME=-o=inception-v1.graph

mvNCCompile -s 12 ${MODEL_FILENAME} ${INPUT_NODE_FLAG} ${OUTPUT_NODE_FLAG} ${OUTPUT_GRAPH_FILENAME}
