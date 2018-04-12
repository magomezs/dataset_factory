import lmdb
import numpy as np
import xmltodict
import cv2
import caffe
from caffe.proto import caffe_pb2

#basic setting
lmdb_file_train_0 = 'train_0_lmdb'
lmdb_file_train_1 = 'train_1_lmdb'
lmdb_file_train_2 = 'train_2_lmdb'
lmdb_file_train_3 = 'train_3_lmdb'
lmdb_file_train_4 = 'train_4_lmdb'
lmdb_file_train_5 = 'train_5_lmdb'
lmdb_file_train_6 = 'train_6_lmdb'
lmdb_file_train_7 = 'train_7_lmdb'
lmdb_file_train_8 = 'train_8_lmdb'
lmdb_file_train_9 = 'train_9_lmdb'
lmdb_file_train_10 = 'train_10_lmdb'
lmdb_file_train_11 = 'train_11_lmdb'
lmdb_file_train_12 = 'train_12_lmdb'
lmdb_file_train_13 = 'train_13_lmdb'
lmdb_file_train_14 = 'train_14_lmdb'
lmdb_file_train_15 = 'train_15_lmdb'
lmdb_file_val_0 = 'val_0_lmdb'
lmdb_file_val_1 = 'val_1_lmdb'
lmdb_file_val_2 = 'val_2_lmdb'
lmdb_file_val_3 = 'val_3_lmdb'
lmdb_file_val_4 = 'val_4_lmdb'
lmdb_file_val_5 = 'val_5_lmdb'
lmdb_file_val_6 = 'val_6_lmdb'
lmdb_file_val_7 = 'val_7_lmdb'
lmdb_file_val_8 = 'val_8_lmdb'
lmdb_file_val_9 = 'val_9_lmdb'
lmdb_file_val_10 = 'val_10_lmdb'
lmdb_file_val_11 = 'val_11_lmdb'
lmdb_file_val_12 = 'val_12_lmdb'
lmdb_file_val_13 = 'val_13_lmdb'
lmdb_file_val_14 = 'val_14_lmdb'
lmdb_file_val_15 = 'val_15_lmdb'

#data files
train_0='../TRACKLET_DATA/train_0.txt'
train_1='../TRACKLET_DATA/train_1.txt'
train_2='../TRACKLET_DATA/train_2.txt'
train_3='../TRACKLET_DATA/train_3.txt'
train_4='../TRACKLET_DATA/train_4.txt'
train_5='../TRACKLET_DATA/train_5.txt'
train_6='../TRACKLET_DATA/train_6.txt'
train_7='../TRACKLET_DATA/train_7.txt'
train_8='../TRACKLET_DATA/train_8.txt'
train_9='../TRACKLET_DATA/train_9.txt'
train_10='../TRACKLET_DATA/train_10.txt'
train_11='../TRACKLET_DATA/train_11.txt'
train_12='../TRACKLET_DATA/train_12.txt'
train_13='../TRACKLET_DATA/train_13.txt'
train_14='../TRACKLET_DATA/train_14.txt'
train_15='../TRACKLET_DATA/train_15.txt'
val_0='../TRACKLET_DATA/val_0.txt'
val_1='../TRACKLET_DATA/val_1.txt'
val_2='../TRACKLET_DATA/val_2.txt'
val_3='../TRACKLET_DATA/val_3.txt'
val_4='../TRACKLET_DATA/val_4.txt'
val_5='../TRACKLET_DATA/val_5.txt'
val_6='../TRACKLET_DATA/val_6.txt'
val_7='../TRACKLET_DATA/val_7.txt'
val_8='../TRACKLET_DATA/val_8.txt'
val_9='../TRACKLET_DATA/val_9.txt'
val_10='../TRACKLET_DATA/val_10.txt'
val_11='../TRACKLET_DATA/val_11.txt'
val_12='../TRACKLET_DATA/val_12.txt'
val_13='../TRACKLET_DATA/val_13.txt'
val_14='../TRACKLET_DATA/val_14.txt'
val_15='../TRACKLET_DATA/val_15.txt'



print 'train_0'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_train_0, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(train_0, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)


print 'train_1'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_train_1, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(train_1, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)


print 'train_2'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_train_2, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(train_2, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)


print 'train_3'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_train_3, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(train_3, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)


print 'train_4'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_train_4, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(train_4, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)


print 'train_5'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_train_5, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(train_5, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)


print 'train_6'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_train_6, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(train_6, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)


print 'train_7'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_train_7, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(train_7, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)


print 'train_8'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_train_8, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(train_8, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)


print 'train_9'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_train_9, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(train_9, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)



print 'train_10'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_train_10, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(train_10, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)



print 'train_11'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_train_11, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(train_11, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)



print 'train_12'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_train_12, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(train_12, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)



print 'train_13'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_train_13, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(train_13, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)


print 'train_14'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_train_14, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(train_14, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)



print 'train_15'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_train_15, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(train_15, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)






print 'val_0'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_val_0, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(val_0, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)




print 'val_1'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_val_1, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(val_1, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)




print 'val_2'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_val_2, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(val_2, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)




print 'val_3'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_val_3, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(val_3, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)




print 'val_4'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_val_4, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(val_4, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)




print 'val_5'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_val_5, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(val_5, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)




print 'val_6'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_val_6, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(val_6, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)




print 'val_7'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_val_7, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(val_7, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)




print 'val_8'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_val_8, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(val_8, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)




print 'val_9'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_val_9, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(val_9, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)




print 'val_10'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_val_10, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(val_10, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)




print 'val_11'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_val_11, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(val_11, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)




print 'val_12'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_val_12, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(val_12, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)




print 'val_13'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_val_13, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(val_13, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)




print 'val_14'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_val_14, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(val_14, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)




print 'val_15'

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file_val_15, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

file = open(val_15, 'r')  
item_id = -1
for line in file:
    item_id += 1
    words = line.split()
    #one descriptor	
    file_path = '../FEATURES/' + words[0] 
    file_path=file_path[:-3]
    file_path=file_path + 'xml'

    #prepare the data and label
    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())
    data = doc['opencv_storage']['feature']['data']
    features = data.split()
    data_array = np.array(features)
    sample = np.zeros((1,1000,1), np.float32)
    sample[0, :,0]=data_array
    label=words[1]

    # save in datum
    datum = caffe.io.array_to_datum(sample, int(label))
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    #write sample
    lmdb_txn.commit()
    lmdb_txn = lmdb_env.begin(write=True)





    # write batch
    #if(item_id + 1) % batch_size == 0:
    #    lmdb_txn.commit()
    #    lmdb_txn = lmdb_env.begin(write=True)

# write last batch
#if (item_id+1) % batch_size != 0:
#    lmdb_txn.commit()
#    print 'last batch'
#    print (item_id + 1)
