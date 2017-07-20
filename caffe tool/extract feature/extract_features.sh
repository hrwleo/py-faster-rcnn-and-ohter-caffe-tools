#!/usr/bin/env sh
#2017-1-19 herongwei
# extract features from path
# N.B. set the path to the imagenet test + val data dirs

TOOLS=build/tools
MODEL_FILE=/home/hrw/caffe/examples/VGG/vgg_face.caffemodel
PROTOTXT_FILE=/home/hrw/caffe/examples/VGG/deploy.prototxt
BLOB_NAME=fc7
INPUT=/home/hrw/caffe/data/val.txt
NUM=1040
FILE_TYPE=leveldb
OUTPUT=/home/hrw/caffe/examples/VGG/test_features_$NUM\_$BLOB_NAME\_$FILE_TYPE
MODE=GPU

if [ ! -f "$MODEL_FILE" ]; then
  echo "Error: MODEL_FILE is not a path to a file: $TMODEL_FILE"
  exit 1
fi

if [ ! -f "$PROTOTXT_FILE" ]; then
  echo "Error: PROTOTXT_FILE is not a path to a file: $PROTOTXT_FILE"
  exit 1
fi

if [ -d $OUTPUT ]; then
    rm -rf $OUTPUT
fi

echo "Extract train features..."

$TOOLS/extract_features \
    $MODEL_FILE \
    $PROTOTXT_FILE \
    $BLOB_NAME \
    $INPUT \
    $OUTPUT \
    $NUM \
    $FILE_TYPE \
    $MODE

echo "Done."
