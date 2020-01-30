#ifndef _data_factory
#define _data_factory

#include <string>

std::string int2str(int);
void create_pair_triplet_data(std::string, int, int, int, float, float);
void create_pair_data(std::string, int, int, int, int, float, float);
void create_triplet_data(std::string, int, int, int, int, float, float);
void create_tracklet_data(std::string, int, int, int, int, int, float);
void create_contiguous_tracklet_data(std::string, int, int, int, int, int, float, float);
void create_reid_tracklet_data(std::string, int, int, int, int, int, int, float, float);
void create_occlusion_tracklet_data(std::string, int, int, int, int, int, int, int, float, float);
void create_intruders_tracklet_data(std::string, int, int, int, int, int, int, float, float);
void create_real_tracklet_data(std::string, int, int, int, int, int, int, int, int, float, float);

#endif
