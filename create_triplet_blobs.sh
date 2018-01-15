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
	BLOBS_ROOT=../TRIPLET_BLOBS
	DATA_FILES=../TRIPLET_DATA

	if [ ! -d "$BLOBS_ROOT" ]; then
		mkdir "$BLOBS_ROOT"
	fi

	echo "Creating train lmdb..."
	GLOG_logtostderr=1 $TOOLS/convert_imageset \
	    --resize_height=128 \
	    --resize_width=64 \
	    --shuffle=false \
	    $SAMPLES \
	    $DATA_FILES/train_an.txt \
	    $BLOBS_ROOT/train_an_lmdb

	echo "Creating val lmdb..."
	GLOG_logtostderr=1 $TOOLS/convert_imageset \
	    --resize_height=128 \
	    --resize_width=64 \
	    --shuffle=false \
	    $SAMPLES \
	    $DATA_FILES/val_an.txt \
	    $BLOBS_ROOT/val_an_lmdb

	echo "Creating train lmdb..."
	GLOG_logtostderr=1 $TOOLS/convert_imageset \
	    --resize_height=128 \
	    --resize_width=64 \
	    --shuffle=false \
	    $SAMPLES \
	    $DATA_FILES/train_p.txt \
	    $BLOBS_ROOT/train_p_lmdb

	echo "Creating val lmdb..."
	GLOG_logtostderr=1 $TOOLS/convert_imageset \
	    --resize_height=128 \
	    --resize_width=64 \
	    --shuffle=false \
	    $SAMPLES \
	    $DATA_FILES/val_p.txt \
	    $BLOBS_ROOT/val_p_lmdb

	echo "Creating train lmdb..."
	GLOG_logtostderr=1 $TOOLS/convert_imageset \
	    --resize_height=128 \
	    --resize_width=64 \
	    --shuffle=false \
	    $SAMPLES \
	    $DATA_FILES/train_n.txt \
	    $BLOBS_ROOT/train_n_lmdb

	echo "Creating val lmdb..."
	GLOG_logtostderr=1 $TOOLS/convert_imageset \
	    --resize_height=128 \
	    --resize_width=64 \
	    --shuffle=false \
	    $SAMPLES \
	    $DATA_FILES/val_n.txt \
	    $BLOBS_ROOT/val_n_lmdb

	echo "Done."
fi


if $INDEPENDENT ; then
	sequences='../sequences_list.txt'
	while read line; do    
		echo $line  

		DATA_FILES=../TRIPLET_DATA/$line
		BLOBS_ROOT=../TRIPLET_BLOBS
		if [ ! -d "$BLOBS_ROOT" ]; then
			mkdir "$BLOBS_ROOT"
		fi

		BLOBS_ROOT=../TRIPLET_BLOBS/$line
		if [ ! -d "$BLOBS_ROOT" ]; then
			mkdir "$BLOBS_ROOT"
		fi


		echo "Creating train lmdb..."
		GLOG_logtostderr=1 $TOOLS/convert_imageset \
		    --resize_height=128 \
		    --resize_width=64 \
		    --shuffle=false \
		    $SAMPLES \
		    $DATA_FILES/${line}_train_an.txt \
		    $BLOBS_ROOT/train_an_lmdb

		echo "Creating val lmdb..."
		GLOG_logtostderr=1 $TOOLS/convert_imageset \
		    --resize_height=128 \
		    --resize_width=64 \
		    --shuffle=false \
		    $SAMPLES \
		    $DATA_FILES/${line}_val_an.txt \
		    $BLOBS_ROOT/val_an_lmdb

		echo "Creating train lmdb..."
		GLOG_logtostderr=1 $TOOLS/convert_imageset \
		    --resize_height=128 \
		    --resize_width=64 \
		    --shuffle=false \
		    $SAMPLES \
		    $DATA_FILES/${line}_train_p.txt \
		    $BLOBS_ROOT/train_p_lmdb

		echo "Creating val lmdb..."
		GLOG_logtostderr=1 $TOOLS/convert_imageset \
		    --resize_height=128 \
		    --resize_width=64 \
		    --shuffle=false \
		    $SAMPLES \
		    $DATA_FILES/${line}_val_p.txt \
		    $BLOBS_ROOT/val_p_lmdb

		echo "Creating train lmdb..."
		GLOG_logtostderr=1 $TOOLS/convert_imageset \
		    --resize_height=128 \
		    --resize_width=64 \
		    --shuffle=false \
		    $SAMPLES \
		    $DATA_FILES/${line}_train_n.txt \
		    $BLOBS_ROOT/train_n_lmdb

		echo "Creating val lmdb..."
		GLOG_logtostderr=1 $TOOLS/convert_imageset \
		    --resize_height=128 \
		    --resize_width=64 \
		    --shuffle=false \
		    $SAMPLES \
		    $DATA_FILES/${line}_val_n.txt \
		    $BLOBS_ROOT/val_n_lmdb

		echo "Done."

	done < $sequences
fi

