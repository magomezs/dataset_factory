#!/usr/bin/env sh

DATASET=../DATASETS/PRID1/
TOOLS=../caffe/build/tools


BLOBS=$DATASET/BLOBS/TEST
DATA=$DATASET/DATA/TEST
SAMPLES=$DATASET/SAMPLES/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=128
  RESIZE_WIDTH=64
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi



if [ ! -d "$SAMPLES" ]; then
  echo "Error: SAMPLES is not a path to a directory: $SAMPLES"
  exit 1
fi

echo "Creating test_a lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=false \
    $SAMPLES \
    $DATA/test_a.txt \
    $BLOBS/test_a_lmdb


echo "Creating test_b lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=false \
    $SAMPLES \
    $DATA/test_b.txt \
    $BLOBS/test_b_lmdb

echo "Done."
