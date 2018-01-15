#!/usr/bin/env sh

GLOBAL=true
INDEPENDENT=false

RESIZE=false


TOOLS=../../caffe/build/tools
SAMPLES=../SAMPLES/


if [ ! -d "$SAMPLES" ]; then
	echo "Error: SAMPLES is not a path to a directory: $SAMPLES"
	echo "Set the SAMPLES variable in create_triplet_blobs.sh to the path" \
	     "where the training samples are stored."
	exit 1
fi

if $GLOBAL ; then
	BLOBS_ROOT=../PAIR_BLOBS
	DATA_FILES=../PAIR_DATA

	if [ ! -d "$BLOBS_ROOT" ]; then
		mkdir "$BLOBS_ROOT"
	fi

	echo "Creating train lmdb..."
	GLOG_logtostderr=1 $TOOLS/convert_imageset \
	    --resize_height=128 \
	    --resize_width=64 \
	    --shuffle=false \
	    $SAMPLES \
	    $DATA_FILES/train_a.txt \
	    $BLOBS_ROOT/train_a_lmdb

	echo "Creating val lmdb..."
	GLOG_logtostderr=1 $TOOLS/convert_imageset \
	    --resize_height=128 \
	    --resize_width=64 \
	    --shuffle=false \
	    $SAMPLES \
	    $DATA_FILES/val_a.txt \
	    $BLOBS_ROOT/val_a_lmdb

	echo "Creating train lmdb..."
	GLOG_logtostderr=1 $TOOLS/convert_imageset \
	    --resize_height=128 \
	    --resize_width=64 \
	    --shuffle=false \
	    $SAMPLES \
	    $DATA_FILES/train_b.txt \
	    $BLOBS_ROOT/train_b_lmdb

	echo "Creating val lmdb..."
	GLOG_logtostderr=1 $TOOLS/convert_imageset \
	    --resize_height=128 \
	    --resize_width=64 \
	    --shuffle=false \
	    $SAMPLES \
	    $DATA_FILES/val_b.txt \
	    $BLOBS_ROOT/val_b_lmdb

	echo "Done."
fi


if $INDEPENDENT ; then
	sequences='../sequences_list.txt'
	while read line; do    
		echo $line  

		DATA_FILES=../PAIR_DATA/$line
		BLOBS_ROOT=../PAIR_BLOBS
		if [ ! -d "$BLOBS_ROOT" ]; then
			mkdir "$BLOBS_ROOT"
		fi

		BLOBS_ROOT=../PAIR_BLOBS/$line
		if [ ! -d "$BLOBS_ROOT" ]; then
			mkdir "$BLOBS_ROOT"
		fi


		echo "Creating train lmdb..."
		GLOG_logtostderr=1 $TOOLS/convert_imageset \
		    --resize_height=128 \
		    --resize_width=64 \
		    --shuffle=false \
		    $SAMPLES \
		    $DATA_FILES/${line}_train_a.txt \
		    $BLOBS_ROOT/train_a_lmdb

		echo "Creating val lmdb..."
		GLOG_logtostderr=1 $TOOLS/convert_imageset \
		    --resize_height=128 \
		    --resize_width=64 \
		    --shuffle=false \
		    $SAMPLES \
		    $DATA_FILES/${line}_val_a.txt \
		    $BLOBS_ROOT/val_a_lmdb

		echo "Creating train lmdb..."
		GLOG_logtostderr=1 $TOOLS/convert_imageset \
		    --resize_height=128 \
		    --resize_width=64 \
		    --shuffle=false \
		    $SAMPLES \
		    $DATA_FILES/${line}_train_b.txt \
		    $BLOBS_ROOT/train_b_lmdb

		echo "Creating val lmdb..."
		GLOG_logtostderr=1 $TOOLS/convert_imageset \
		    --resize_height=128 \
		    --resize_width=64 \
		    --shuffle=false \
		    $SAMPLES \
		    $DATA_FILES/${line}_val_b.txt \
		    $BLOBS_ROOT/val_b_lmdb

		echo "Done."

	done < $sequences
fi

