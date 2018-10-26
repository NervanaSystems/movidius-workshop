#!/bin/bash
cd "$(pwd)/inception_v1/images/"
for i in $(ls); do curl -X POST -H "Content-Type: multipart/form-data" -F "the_file=@${i}" http://10.42.0.1:5000/classify_picture; done
