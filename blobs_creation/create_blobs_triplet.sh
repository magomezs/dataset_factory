#!/usr/bin/env sh

DATASET=../DATASETS/PRID1/
TOOLS=../caffe/build/tools


BLOBS=$DATASET/BLOBS/TRIPLET
DATA=$DATASET/DATA/TRIPLET
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
    $DATA/train_an.txt \
    $BLOBS/train_an_lmdb


GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=false \
    $SAMPLES \
    $DATA/train_p.txt \
    $BLOBS/train_p_lmdb


GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=false \
    $SAMPLES \
    $DATA/train_n.txt \
    $BLOBS/train_n_lmdb


echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=false \
    $SAMPLES \
    $DATA/val_an.txt \
    $BLOBS/val_an_lmdb

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=false \
    $SAMPLES \
    $DATA/val_p.txt \
    $BLOBS/val_p_lmdb

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=false \
    $SAMPLES \
    $DATA/val_n.txt \
    $BLOBS/val_n_lmdb

echo "Done."
