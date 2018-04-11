#!/usr/bin/env python
import os
import data_factory

data_factory.get_samples(0.5, True, True); 
data_factory.get_pair_data_files(True, True, 1000000, 50000, 16, 1, 0.9); 
data_factory.get_triplet_data_files(True, True, 1000000, 50000, 16, 1, 0.9); 
data_factory.get_tracklet_data_files(True, True, 1000000, 50000, 16, 16, 1, 0.9); 


os.system("sudo chmod +x create_pair_blobs.sh")
os.system("./create_pair_blobs.sh")

os.system("sudo chmod +x create_triplet_blobs.sh")
os.system("./create_triplet_blobs.sh")

os.system("sudo chmod +x create_tracklet_blobs.sh")
os.system("./create_tracklet_blobs.sh")

