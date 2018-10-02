#! /bin/bash

WEIGHTS_FILENAME=inception_v1_2016_08_28.tar.gz

wget http://download.tensorflow.org/models/${WEIGHTS_FILENAME} && tar zxf ${WEIGHTS_FILENAME} && rm ${WEIGHTS_FILENAME}
