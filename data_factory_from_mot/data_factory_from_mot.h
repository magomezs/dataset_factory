#ifndef _data_factory
#define _data_factory

#include <string>

std::string int2str(int);
void get_samples(std::string, float);
void create_pair_data(std::string, int, int, int, int, float);
void create_triplet_data(std::string, int, int, int, int, float);
void create_tracklet_data(std::string, int, int, int, int, int, float);

#endif
