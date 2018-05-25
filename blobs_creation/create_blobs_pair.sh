#!/usr/bin/env sh

DATASET=dataset_directory
TOOLS=../caffe/build/tools

BLOBS=$DATASET/BLOBS/PAIR
DATA=$DATASET/DATA/PAIR/   
SAMPLES=$DATASET/SAMPLES/


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
  echo "Set the SAMPLES variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi


echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=false \
    $SAMPLES \
    $DATA/train_a.txt \
    $BLOBS/train_a_lmdb


GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=false \
    $SAMPLES \
    $DATA/train_b.txt \
    $BLOBS/train_b_lmdb


echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=false \
    $SAMPLES \
    $DATA/val_a.txt \
    $BLOBS/val_a_lmdb


GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=false \
    $SAMPLES \
    $DATA/val_b.txt \
    $BLOBS/val_b_lmdb

echo "Done."
