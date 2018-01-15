#!/usr/bin/env sh

GLOBAL=true
INDEPENDENT=false
DEPTH=16

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
	BLOBS_ROOT=../TRACKLET_BLOBS
	DATA_FILES=../TRACKLET_DATA

	if [ ! -d "$BLOBS_ROOT" ]; then
		mkdir "$BLOBS_ROOT"
	fi
	
        n=0
	while [ $n -lt $DEPTH ]; do
		echo $n
		echo "Creating train lmdb..."
		GLOG_logtostderr=1 $TOOLS/convert_imageset \
		    --resize_height=128 \
		    --resize_width=64 \
		    --shuffle=false \
		    $SAMPLES \
		    $DATA_FILES/train_${n}.txt \
		    $BLOBS_ROOT/train_${n}_lmdb

		echo "Creating val lmdb..."
		GLOG_logtostderr=1 $TOOLS/convert_imageset \
		    --resize_height=128 \
		    --resize_width=64 \
		    --shuffle=false \
		    $SAMPLES \
		    $DATA_FILES/val_${n}.txt \
		    $BLOBS_ROOT/val_${n}_lmdb

		echo "Done."
		n=$((n+1))
        done
fi


if $INDEPENDENT ; then
	sequences='../sequences_list.txt'
	while read line; do    
		echo $line  

		DATA_FILES=../TRACKLET_DATA/$line
		BLOBS_ROOT=../TRACKLET_BLOBS
		if [ ! -d "$BLOBS_ROOT" ]; then
			mkdir "$BLOBS_ROOT"
		fi

		BLOBS_ROOT=../TRACKLET_BLOBS/$line
		if [ ! -d "$BLOBS_ROOT" ]; then
			mkdir "$BLOBS_ROOT"
		fi

		n=0
		while [ $n -lt $DEPTH ]; do
			echo $n
			echo "Creating train lmdb..."
			GLOG_logtostderr=1 $TOOLS/convert_imageset \
			    --resize_height=128 \
			    --resize_width=64 \
			    --shuffle=false \
			    $SAMPLES \
			    $DATA_FILES/${line}_train_${n}.txt \
			    $BLOBS_ROOT/train_${n}_lmdb

			echo "Creating val lmdb..."
			GLOG_logtostderr=1 $TOOLS/convert_imageset \
			    --resize_height=128 \
			    --resize_width=64 \
			    --shuffle=false \
			    $SAMPLES \
			    $DATA_FILES/${line}_val_${n}.txt \
			    $BLOBS_ROOT/val_${n}_lmdb

			echo "Done."
			n=$((n+1))
		done

	done < $sequences
fi
