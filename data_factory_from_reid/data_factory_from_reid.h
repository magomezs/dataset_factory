#ifndef _dataset_factory_from_reid
#define _dataset_factory_from_reid

#include <string>

using namespace std;

void get_samples(string, int, int);
void train_val_test_division(string, int, int, int, int, int, int, int);
void create_pair_data(string, int, int, int, int);
void create_triplet_data(string, int, int);
void create_triplet_data_fixed_cam(string, int, int);
void create_test_data(string);

#endif
