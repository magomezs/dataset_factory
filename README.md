# dataset_factory

This repository contains two groups of functions in C++

- data_factory_from_reid: generation of pairs and triplets from Re-Identification datasets with the division of the samples in train, validation and test sets, according to the protocol described by described in [3].

- data_factory_from_mot: generation of pairs, triplets and tracklets from Multi-object tracking datasets with the division of the samples in train, validation and test sets, according to the protocol described by described in [3].


The outputs are data txt files with labels, suitable for blobs creation to train deep networks with caffe.

<br />

# Example of how to use data_factory_from_reid 
This is an example of how to use data_factory_from_reid with PRID2011[1] and ViPER datasets[2].

<br />

string prid= "prid_dataset_directory" <br />
get_samples(prid, 7,4);<br />
train_val_test_division(prid, 100, 100, 100, 10, 100, 649, 100);<br />
create_pair_data(prid, 100000, 10000, 1,4);<br />
create_triplet_data_fixed_cam(prid, 50000, 5000);<br />
create_triplet_data(prid, 50000, 5000);<br />
create_test_data(prid);<br />
<br />

string viper= "viper_dataset_directory"<br />
get_samples(viper, 0, 3);<br />
train_val_test_division(viper, 316, 316, 316, 10, 316, 316, 316);<br />
create_pair_data(viper, 100000, 10000, 1,4);<br />
create_triplet_data(viper, 50000, 5000);<br />
create_triplet_data_fixed_cam(viper, 50000, 5000);<br />
create_test_data(viper);<br />
<br />


NOTE:be carefull with PRID samples whose identification number is higher than 200, because different people in cam a and b are labbelled with the same number, from id 200. Alternative solution: remove samples with ID higher than 200 in cam_a set, they are not neccesarry in the training and test described in [3].

<br />

[1]Person Re-Identification by Descriptive and Discriminative Classification, Martin Hirzer, Csaba Beleznai, Peter M. Roth and Horst Bischof, In Proc. Scandinavian Conference on Image Analysis (SCIA), 2011

[2]D. Gray, and H. Tao, "Viewpoint Invariant Pedestrian Recognition with an Ensemble of Localized Features," in Proc. European Conference on Computer Vision (ECCV), 2008.

[3]Hirzer, M., Beleznai, C., Roth, P. M., and Bischof, H. (2011). Person re-identification by descriptive and
discriminative classification. In Scandinavian conference on Image analysis, pages 91–102. Springer.

