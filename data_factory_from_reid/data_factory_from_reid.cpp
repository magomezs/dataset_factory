#include "data_factory_from_reid.h"
#include <stdio.h>
#include <string>
#include <time.h>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>


using namespace std;

void get_samples(string dataset_folder, int pos , int len){
    //SAMPLES FOLDERS
    char samples_folder[150], cam_a_folder[150], cam_b_folder[150];
    strcpy(samples_folder, dataset_folder.c_str());
    strcat(samples_folder, "/SAMPLES");
    strcpy(cam_a_folder, samples_folder);
    strcat(cam_a_folder, "/cam_a");
    strcpy(cam_b_folder, samples_folder);
    strcat(cam_b_folder, "/cam_b");

    //LIST OF FILES IN cam_a (random)
    char file_cam_a_name[150];
    strcpy(file_cam_a_name, dataset_folder.c_str());
    strcat(file_cam_a_name, "/DATA/cam_a.txt");
    ofstream file_cam_a(file_cam_a_name);
    if (!file_cam_a.is_open())
        cout<<"1. can not open "<<file_cam_a_name<<endl;
    DIR *dir_a;
    struct dirent *ent_a;
    if ((dir_a = opendir (cam_a_folder)) != NULL) {
      while ((ent_a = readdir (dir_a)) != NULL) {
          string file= ent_a->d_name;
          if(file!="." && file!=".."){
              string label = file.substr (pos,len);
              file_cam_a<<"cam_a/"<<file<<" "<<label<<"\n";
          }
      }
      closedir (dir_a);
    } else {
      perror ("2. could not open directory");
    }
    file_cam_a.close();


    //LIST OF FILES IN cam_b (random)
    char file_cam_b_name[150];
    strcpy(file_cam_b_name, dataset_folder.c_str());
    strcat(file_cam_b_name, "/DATA/cam_b.txt");
    ofstream file_cam_b(file_cam_b_name);
    if (!file_cam_b.is_open())
        cout<<"3. can not open "<<file_cam_b_name<<endl;
    DIR *dir_b;
    struct dirent *ent_b;
    if ((dir_b = opendir (cam_b_folder)) != NULL) {
      while ((ent_b = readdir (dir_b)) != NULL) {
          string file= ent_b->d_name;
          if(file!="." && file!=".."){
              string label = file.substr (pos,len);
              file_cam_b<<"cam_b/"<<file<<" "<<label<<"\n";
          }

      }
      closedir (dir_b);
    } else {
      perror ("4. could not open directory");
    }
    file_cam_b.close();

}


void train_val_test_division(string dataset_folder, int a_samples_for_training, int b_samples_for_training,  int repeated_samples_for_training, int val_percentage, int a_samples_for_testing, int b_samples_for_testing,  int repeated_samples_for_testing){
    int ratio;
    if (val_percentage==0){
        ratio=100+a_samples_for_training+b_samples_for_training;
    }else{
        ratio=int(100/val_percentage);
    }

    char file_cam_a_name[150], file_cam_a_train_name[150], file_cam_a_test_name[150], file_cam_a_val_name[150], file_cam_b_name[150], file_cam_b_train_name[150], file_cam_b_test_name[150], file_cam_b_val_name[150];

    strcpy(file_cam_a_name, dataset_folder.c_str());
    strcat(file_cam_a_name, "/DATA/cam_a.txt");
    ifstream file_cam_a(file_cam_a_name);
    if (!file_cam_a.is_open())
        cout<<"1. can not open "<<file_cam_a_name<<endl;

    strcpy(file_cam_a_train_name, dataset_folder.c_str());
    strcat(file_cam_a_train_name, "/DATA/cam_a_train.txt");
    ofstream file_cam_a_train(file_cam_a_train_name);
    if (!file_cam_a_train.is_open())
        cout<<"2. can not open "<<file_cam_a_train_name<<endl;

    strcpy(file_cam_a_val_name, dataset_folder.c_str());
    strcat(file_cam_a_val_name, "/DATA/cam_a_val.txt");
    ofstream file_cam_a_val(file_cam_a_val_name);
    if (!file_cam_a_val.is_open())
        cout<<"3. can not open "<<file_cam_a_val_name<<endl;

    strcpy(file_cam_a_test_name, dataset_folder.c_str());
    strcat(file_cam_a_test_name, "/DATA/cam_a_test.txt");
    ofstream file_cam_a_test(file_cam_a_test_name);
    if (!file_cam_a_test.is_open())
        cout<<"4. can not open "<<file_cam_a_test_name<<endl;


    strcpy(file_cam_b_name, dataset_folder.c_str());
    strcat(file_cam_b_name, "/DATA/cam_b.txt");
    ifstream file_cam_b(file_cam_b_name);
    if (!file_cam_b.is_open())
        cout<<"5. can not open "<<file_cam_b_name<<endl;

    strcpy(file_cam_b_train_name, dataset_folder.c_str());
    strcat(file_cam_b_train_name, "/DATA/cam_b_train.txt");
    ofstream file_cam_b_train(file_cam_b_train_name);
    if (!file_cam_b_train.is_open())
        cout<<"6. can not open "<<file_cam_b_train_name<<endl;

    strcpy(file_cam_b_val_name, dataset_folder.c_str());
    strcat(file_cam_b_val_name, "/DATA/cam_b_val.txt");
    ofstream file_cam_b_val(file_cam_b_val_name);
    if (!file_cam_b_val.is_open())
        cout<<"7. can not open "<<file_cam_b_val_name<<endl;

    strcpy(file_cam_b_test_name, dataset_folder.c_str());
    strcat(file_cam_b_test_name, "/DATA/cam_b_test.txt");
    ofstream file_cam_b_test(file_cam_b_test_name);
    if (!file_cam_b_test.is_open())
        cout<<"8. can not open "<<file_cam_b_test_name<<endl;


    string linea, sample_a, sample_b, label_a, label_b;

    int count_a_tr=1;
    int count_a_ts=1;
    while(count_a_tr<=a_samples_for_training || count_a_ts<=a_samples_for_testing){
        file_cam_a>> sample_a >> label_a;
        while(getline (file_cam_a,linea)){
            if(count_a_tr <= a_samples_for_training){//search samples for training or validation
                if(count_a_tr%ratio==0){//validation
                    if(count_a_tr<=repeated_samples_for_training){
                        bool is_repeated=false;
                        ifstream file_cam_b_aux(file_cam_b_name);
                        if (!file_cam_b_aux.is_open())
                            cout<<"can not open "<<file_cam_b_name<<endl;
                        file_cam_b_aux>> sample_b >> label_b;
                        while(getline (file_cam_b_aux,linea)){
                            if(label_a==label_b){
                                is_repeated=true;
                            }
                            file_cam_b_aux>> sample_b >> label_b;
                        }
                        file_cam_b_aux.close();
                        if(is_repeated){
                            file_cam_a_val<<sample_a<<" "<<label_a<<"\n";
                            count_a_tr++;
                        }else{
                            if(count_a_ts<=a_samples_for_testing){
                                if(count_a_ts<=repeated_samples_for_testing){
                                    bool is_repeated=false;
                                    ifstream file_cam_b_aux(file_cam_b_name);
                                    if (!file_cam_b_aux.is_open())
                                        cout<<"can not open "<<file_cam_b_name<<endl;
                                    file_cam_b_aux>> sample_b >> label_b;
                                    while(getline (file_cam_b_aux,linea)){
                                        if(label_a==label_b){
                                            is_repeated=true;
                                        }
                                        file_cam_b_aux>> sample_b >> label_b;
                                    }
                                    file_cam_b_aux.close();
                                    if(is_repeated){
                                        file_cam_a_test<<sample_a<<" "<<label_a<<"\n";
                                        count_a_ts++;
                                    }
                                }else{
                                    file_cam_a_test<<sample_a<<" "<<label_a<<"\n";
                                    count_a_ts++;
                                }
                            }
                        }
                    }else{
                        file_cam_a_val<<sample_a<<" "<<label_a<<"\n";
                        count_a_tr++;
                    }
                }else{//training
                    if(count_a_tr<=repeated_samples_for_training){
                        bool is_repeated=false;
                        ifstream file_cam_b_aux(file_cam_b_name);
                        if (!file_cam_b_aux.is_open())
                            cout<<"can not open "<<file_cam_b_name<<endl;
                        file_cam_b_aux>> sample_b >> label_b;
                        while(getline (file_cam_b_aux,linea)){
                            if(label_a==label_b){
                                is_repeated=true;
                            }
                            file_cam_b_aux>> sample_b >> label_b;
                        }
                        file_cam_b_aux.close();
                        if(is_repeated){
                            file_cam_a_train<<sample_a<<" "<<label_a<<"\n";
                            count_a_tr++;
                        }else{
                            if(count_a_ts<=a_samples_for_testing){
                                if(count_a_ts<=repeated_samples_for_testing){

                                    bool is_repeated=false;
                                    ifstream file_cam_b_aux(file_cam_b_name);
                                    if (!file_cam_b_aux.is_open())
                                        cout<<"can not open "<<file_cam_b_name<<endl;
                                    file_cam_b_aux>> sample_b >> label_b;
                                    while(getline (file_cam_b_aux,linea)){
                                        if(label_a==label_b){
                                            is_repeated=true;
                                        }
                                        file_cam_b_aux>> sample_b >> label_b;
                                    }
                                    file_cam_b_aux.close();
                                    if(is_repeated){
                                        file_cam_a_test<<sample_a<<" "<<label_a<<"\n";
                                        count_a_ts++;
                                    }
                                }else{
                                    file_cam_a_test<<sample_a<<" "<<label_a<<"\n";
                                    count_a_ts++;
                                }
                            }

                        }
                    }else{
                        file_cam_a_train<<sample_a<<" "<<label_a<<"\n";
                        count_a_tr++;
                    }
                }
            }else{//search samples for testing
                if(count_a_ts<=a_samples_for_testing){
                    if(count_a_ts<=repeated_samples_for_testing){

                        bool is_repeated=false;
                        ifstream file_cam_b_aux(file_cam_b_name);
                        if (!file_cam_b_aux.is_open())
                            cout<<"can not open "<<file_cam_b_name<<endl;
                        file_cam_b_aux>> sample_b >> label_b;
                        while(getline (file_cam_b_aux,linea)){
                            if(label_a==label_b){
                                is_repeated=true;
                            }
                            file_cam_b_aux>> sample_b >> label_b;
                        }
                        file_cam_b_aux.close();
                        if(is_repeated){
                            file_cam_a_test<<sample_a<<" "<<label_a<<"\n";
                            count_a_ts++;
                        }
                    }else{
                        file_cam_a_test<<sample_a<<" "<<label_a<<"\n";
                        count_a_ts++;
                    }
                }
            }
            file_cam_a>> sample_a >> label_a;
        }
        file_cam_a.close();
        file_cam_a.open(file_cam_a_name);
        if (!file_cam_a.is_open())
            cout<<"can not open "<<file_cam_a_name<<endl;

    }

    file_cam_a_train.close();
    file_cam_a_test.close();
    file_cam_a_val.close();
    file_cam_a.close();





    int count_b_tr=1;
    int count_b_ts=1;
    while(count_b_tr<=b_samples_for_training || count_b_ts<=b_samples_for_testing){
        file_cam_b>> sample_b >> label_b;
        while(getline (file_cam_b,linea)){
            if(count_b_tr <= b_samples_for_training){//search samples for training
                if(count_b_tr<=repeated_samples_for_training){
                    bool is_repeated=false;
                    ifstream file_cam_a_aux(file_cam_a_train_name);
                    if (!file_cam_a_aux.is_open())
                        cout<<"can not open "<<file_cam_a_train_name<<endl;
                    file_cam_a_aux>> sample_a >> label_a;
                    while(getline (file_cam_a_aux,linea)){
                        if(label_a==label_b){
                            is_repeated=true;
                        }
                        file_cam_a_aux>> sample_a >> label_a;
                    }
                    file_cam_a_aux.close();
                    if(is_repeated){
                        file_cam_b_train<<sample_b<<" "<<label_b<<"\n";
                        count_b_tr++;
                    }else{
                        file_cam_a_aux.open(file_cam_a_val_name);
                        if (!file_cam_a_aux.is_open())
                            cout<<"can not open "<<file_cam_a_val_name<<endl;
                        file_cam_a_aux>> sample_a >> label_a;
                        while(getline (file_cam_a_aux,linea)){
                            if(label_a==label_b){
                                is_repeated=true;
                            }
                            file_cam_a_aux>> sample_a >> label_a;
                        }
                        file_cam_a_aux.close();
                        if(is_repeated){
                            file_cam_b_val<<sample_b<<" "<<label_b<<"\n";
                            count_b_tr++;
                        }else{
                            if(count_b_ts<=b_samples_for_testing){
                                if(count_b_ts<=repeated_samples_for_testing){
                                    bool is_repeated=false;
                                    ifstream file_cam_a_aux(file_cam_a_test_name);
                                    if (!file_cam_a_aux.is_open())
                                        cout<<"can not open "<<file_cam_a_test_name<<endl;
                                    file_cam_a_aux>> sample_a >> label_a;
                                    while(getline (file_cam_a_aux,linea)){
                                        if(label_a==label_b){
                                            is_repeated=true;
                                        }
                                        file_cam_a_aux>> sample_a >> label_a;
                                    }
                                    file_cam_a_aux.close();
                                    if(is_repeated){
                                        file_cam_b_test<<sample_b<<" "<<label_b<<"\n";
                                        count_b_ts++;
                                    }
                                }else{
                                    file_cam_b_test<<sample_b<<" "<<label_b<<"\n";
                                    count_b_ts++;
                                }
                            }

                        }
                    }
                }else{
                    if(count_b_tr%ratio==0){
                        file_cam_b_val<<sample_b<<" "<<label_b<<"\n";
                    }else{
                        file_cam_b_train<<sample_b<<" "<<label_b<<"\n";
                    }
                    count_b_tr++;
                }

            }else{//searching samples for testing
                if(count_b_ts<=b_samples_for_testing){
                    if(count_b_ts<=repeated_samples_for_testing){
                        bool is_repeated=false;
                        ifstream file_cam_a_aux(file_cam_a_test_name);
                        if (!file_cam_a_aux.is_open())
                            cout<<"can not open "<<file_cam_a_test_name<<endl;
                        file_cam_a_aux>> sample_a >> label_a;
                        while(getline (file_cam_a_aux,linea)){
                            if(label_a==label_b){
                                is_repeated=true;
                            }
                            file_cam_a_aux>> sample_a >> label_a;
                        }
                        file_cam_a_aux.close();
                        if(is_repeated){
                            file_cam_b_test<<sample_b<<" "<<label_b<<"\n";
                            count_b_ts++;
                        }
                    }else{
                        file_cam_b_test<<sample_b<<" "<<label_b<<"\n";
                        count_b_ts++;
                    }
                }
            }
            file_cam_b>> sample_b >> label_b;
        }
        file_cam_b.close();
        file_cam_b.open(file_cam_b_name);
        if (!file_cam_b.is_open())
            cout<<"can not open "<<file_cam_b_name<<endl;
    }

    file_cam_b.close();
    file_cam_b_train.close();
    file_cam_b_test.close();
    file_cam_b_val.close();

}


void create_pair_data(string dataset_folder, int trainset_size, int valset_size, int np, int nn){
    //DATA FOLDER
    char data_folder[150];
    strcpy(data_folder, dataset_folder.c_str());
    strcat(data_folder, "/UMBALANCE_DATA");
    int e=mkdir(data_folder, ACCESSPERMS);

    //DATA PAIRS FOLDER
    char data_pair_folder[150];
    strcpy(data_pair_folder, data_folder);
    strcat(data_pair_folder, "/PAIR");
    int ePAIR=mkdir(data_pair_folder, ACCESSPERMS);



    //train data sets----------------------------------------------------------------------------------------------------

    char file_cam_a_train_name[150], file_cam_b_train_name[150], file_a_train_name[150], file_b_train_name[150];

    strcpy(file_cam_a_train_name, data_folder);
    strcat(file_cam_a_train_name, "/cam_a_train.txt");
    ifstream file_cam_a_train(file_cam_a_train_name);
    if (!file_cam_a_train.is_open())
        cout<<"1. can not open "<<file_cam_a_train_name<<endl;

    strcpy(file_cam_b_train_name, data_folder);
    strcat(file_cam_b_train_name, "/cam_b_train.txt");
    ifstream file_cam_b_train(file_cam_b_train_name);
    if (!file_cam_b_train.is_open())
        cout<<"2. can not open "<<file_cam_b_train_name<<endl;

    strcpy(file_a_train_name, data_pair_folder);
    strcat(file_a_train_name, "/train_a.txt");
    ofstream file_train_a(file_a_train_name);
    if (!file_train_a.is_open())
        cout<<"3. can not open "<<file_a_train_name<<endl;

    strcpy(file_b_train_name, data_pair_folder);
    strcat(file_b_train_name, "/train_b.txt");
    ofstream file_train_b(file_b_train_name);
    if (!file_train_b.is_open())
        cout<<"4. can not open "<<file_b_train_name<<endl;


    //max id in train_a
    string line;
    int max_id_a=0;
    while ( std::getline (file_cam_a_train,line))//
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        if(id>max_id_a)
            max_id_a=id;
    }
    file_cam_a_train.close();

    //max id in train_b
    int max_id_b=0;
    while ( std::getline (file_cam_b_train,line))//
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        if(id>max_id_b)
            max_id_b=id;
    }
    file_cam_b_train.close();


    file_cam_a_train.open(file_cam_a_train_name);
    if (!file_cam_a_train.is_open())
        cout<<"can not open "<<file_cam_a_train_name<<endl;

    file_cam_b_train.open(file_cam_b_train_name);
    if (!file_cam_b_train.is_open())
        cout<<"can not open "<<file_cam_b_train_name<<endl;


    //vectors WITH THE SAMPLES NAMES train A and B
    vector<string> samples_train_a, samples_train_b;
    samples_train_a.resize(max_id_a);
    samples_train_b.resize(max_id_b);
    while ( std::getline (file_cam_a_train,line))
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        samples_train_a[id-1]=line;
    }
    file_cam_a_train.close();
    while ( std::getline (file_cam_b_train,line))
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        samples_train_b[id-1]=line;
    }
    file_cam_b_train.close();

    //positive and negative matrixes
    vector< vector<string> > positive_train_pairs, negative_train_pairs;
    positive_train_pairs.resize(max_id_a);
    negative_train_pairs.resize(max_id_a * max_id_b);
    for(int i=0; i<max_id_a; i++){
        string sample_a=samples_train_a[i];
        string sample_b=samples_train_b[i];
        if(!sample_a.empty() && !sample_b.empty()){
           vector<string> pair;
           pair.resize(2);
           pair[0]=sample_a;
           pair[1]=sample_b;
           positive_train_pairs[i]=pair;
        }
    }
    for(int a=0; a<max_id_a; a++){
        string sample_a=samples_train_a[a];
        if(!sample_a.empty()){
           for(int b=0; b<max_id_b; b++){
                string sample_b=samples_train_b[b];
                if(!sample_b.empty() && a!=b){
                    vector<string> pair;
                    pair.push_back(sample_a);
                    pair.push_back(sample_b);
                    int idx=a*max_id_b+b;
                    negative_train_pairs[idx]=pair;
                }
           }
        }
    }


    srand(time(NULL));
    int n_pair=0;
    while(n_pair<trainset_size){

        cout<<n_pair<<endl;
        int p=0;
        while(p<np){
            int randIndex= rand()%min(max_id_a, max_id_b);
            vector<string> pair=positive_train_pairs[randIndex];
            if(!pair.empty()){
                string sample_a=pair[0];
                string sample_b=pair[1];
                file_train_a<<sample_a<<"\n";
                file_train_b<<sample_b<<"\n";
                p++;
                n_pair++;
                cout<<n_pair<<endl;
            }
        }
        int n=0;
        while(n<nn){
            int randIndex= rand()%(max_id_a*max_id_b);
            vector<string> pair=negative_train_pairs[randIndex];
            if(!pair.empty()){
                string sample_a=pair[0];
                string sample_b=pair[1];
                file_train_a<<sample_a<<"\n";
                file_train_b<<sample_b<<"\n";
                n++;
                n_pair++;
                cout<<n_pair<<endl;
            }
        }
        cout<<n_pair<<endl;
    }
    file_train_a.close();
    file_train_b.close();



    //val data sets----------------------------------------------------------------------------------------------------
    char file_cam_a_val_name[150], file_cam_b_val_name[150], file_a_val_name[150], file_b_val_name[150];

    strcpy(file_cam_a_val_name, data_folder);
    strcat(file_cam_a_val_name, "/cam_a_val.txt");
    ifstream file_cam_a_val(file_cam_a_val_name);
    if (!file_cam_a_val.is_open())
        cout<<"1. can not open "<<file_cam_a_val_name<<endl;

    strcpy(file_cam_b_val_name, data_folder);
    strcat(file_cam_b_val_name, "/cam_b_val.txt");
    ifstream file_cam_b_val(file_cam_b_val_name);
    if (!file_cam_b_val.is_open())
        cout<<"2. can not open "<<file_cam_b_val_name<<endl;

    strcpy(file_a_val_name, data_pair_folder);
    strcat(file_a_val_name, "/val_a.txt");
    ofstream file_val_a(file_a_val_name);
    if (!file_val_a.is_open())
        cout<<"3. can not open "<<file_a_val_name<<endl;

    strcpy(file_b_val_name, data_pair_folder);
    strcat(file_b_val_name, "/val_b.txt");
    ofstream file_val_b(file_b_val_name);
    if (!file_val_b.is_open())
        cout<<"4. can not open "<<file_b_val_name<<endl;

    //max id in val_a
    max_id_a=0;
    while ( std::getline (file_cam_a_val,line))//
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        if(id>max_id_a)
            max_id_a=id;
    }
    file_cam_a_val.close();

    //max id in val_b
    max_id_b=0;
    while ( std::getline (file_cam_b_val,line))//
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        if(id>max_id_b)
            max_id_b=id;
    }
    file_cam_b_val.close();


    file_cam_a_val.open(file_cam_a_val_name);
    if (!file_cam_a_val.is_open())
        cout<<"can not open "<<file_cam_a_val_name<<endl;

    file_cam_b_val.open(file_cam_b_val_name);
    if (!file_cam_b_val.is_open())
        cout<<"can not open "<<file_cam_b_val_name<<endl;


    //vectors WITH THE SAMPLES NAMES val A and B
    vector<string> samples_val_a, samples_val_b;
    samples_val_a.resize(max_id_a);
    samples_val_b.resize(max_id_b);
    while ( std::getline (file_cam_a_val,line))
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        samples_val_a[id-1]=line;
    }
    file_cam_a_val.close();
    while ( std::getline (file_cam_b_val,line))
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        samples_val_b[id-1]=line;
    }
    file_cam_b_val.close();

    //positive and negative matrixes
    vector< vector<string> > positive_val_pairs, negative_val_pairs;
    positive_val_pairs.resize(min(max_id_a, max_id_b));
    negative_val_pairs.resize(max_id_a * max_id_b);
    for(int i=0; i<max_id_a; i++){
        string sample_a=samples_val_a[i];
        string sample_b=samples_val_b[i];
        if(!sample_a.empty() && !sample_b.empty()){
           vector<string> pair;
           pair.push_back(sample_a);
           pair.push_back(sample_b);
           positive_val_pairs[i]=pair;
        }
    }
    for(int a=0; a<max_id_a; a++){
        string sample_a=samples_val_a[a];
        if(!sample_a.empty()){
           for(int b=0; b<max_id_b; b++){
                string sample_b=samples_val_b[b];
                if(!sample_b.empty() && a!=b){
                    vector<string> pair;
                    pair.push_back(sample_a);
                    pair.push_back(sample_b);
                    negative_val_pairs[a*max_id_b+b]=pair;
                }
           }
        }
    }


    srand(time(NULL));
    n_pair=0;
    while(n_pair<valset_size){
        int p=0;
        while(p<np){
            int randIndex= rand()%min(max_id_a, max_id_b);
            vector<string> pair=positive_val_pairs[randIndex];
            if(!pair.empty()){
                string sample_a=pair[0];
                string sample_b=pair[1];
                file_val_a<<sample_a<<"\n";
                file_val_b<<sample_b<<"\n";
                p++;
                n_pair++;
            }
        }
        int n=0;
        while(n<nn){
            int randIndex= rand()%(max_id_a*max_id_b);
            vector<string> pair=negative_val_pairs[randIndex];
            if(!pair.empty()){
                string sample_a=pair[0];
                string sample_b=pair[1];
                file_val_a<<sample_a<<"\n";
                file_val_b<<sample_b<<"\n";
                n++;
                n_pair++;
            }
        }
        cout<<n_pair<<endl;
    }
    file_val_a.close();
    file_val_b.close();

}


void create_triplet_data_fixed_cam(string dataset_folder, int trainset_size, int valset_size){
    //DATA FOLDER
    char data_folder[150];
    strcpy(data_folder, dataset_folder.c_str());
    strcat(data_folder, "/DATA");
    int e=mkdir(data_folder, ACCESSPERMS);

    //DATA TRIPLET FOLDER
    char data_triplet_folder[150];
    strcpy(data_triplet_folder, data_folder);
    strcat(data_triplet_folder, "/TRIPLET");
    int etri=mkdir(data_triplet_folder, ACCESSPERMS);





    //train data sets----------------------------------------------------------------------------------------------------

    char file_cam_a_train_name[150], file_cam_b_train_name[150], file_an_train_name[150], file_p_train_name[150], file_n_train_name[150];

    strcpy(file_cam_a_train_name, data_folder);
    strcat(file_cam_a_train_name, "/cam_a_train.txt");
    ifstream file_cam_a_train(file_cam_a_train_name);
    if (!file_cam_a_train.is_open())
        cout<<"1. can not open "<<file_cam_a_train_name<<endl;

    strcpy(file_cam_b_train_name, data_folder);
    strcat(file_cam_b_train_name, "/cam_b_train.txt");
    ifstream file_cam_b_train(file_cam_b_train_name);
    if (!file_cam_b_train.is_open())
        cout<<"2. can not open "<<file_cam_b_train_name<<endl;

    strcpy(file_an_train_name, data_triplet_folder);
    strcat(file_an_train_name, "/train_an_fixed.txt");
    ofstream file_train_an(file_an_train_name);
    if (!file_train_an.is_open())
        cout<<"3. can not open "<<file_an_train_name<<endl;

    strcpy(file_p_train_name, data_triplet_folder);
    strcat(file_p_train_name, "/train_p_fixed.txt");
    ofstream file_train_p(file_p_train_name);
    if (!file_train_p.is_open())
        cout<<"4. can not open "<<file_p_train_name<<endl;

    strcpy(file_n_train_name, data_triplet_folder);
    strcat(file_n_train_name, "/train_n_fixed.txt");
    ofstream file_train_n(file_n_train_name);
    if (!file_train_n.is_open())
        cout<<"5. can not open "<<file_n_train_name<<endl;


    //max id in train_a
    string line;
    int max_id_a=0;
    while ( std::getline (file_cam_a_train,line))//
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        if(id>max_id_a)
            max_id_a=id;
    }
    file_cam_a_train.close();

    //max id in train_b
    int max_id_b=0;
    while ( std::getline (file_cam_b_train,line))//
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        if(id>max_id_b)
            max_id_b=id;
    }
    file_cam_b_train.close();


    file_cam_a_train.open(file_cam_a_train_name);
    if (!file_cam_a_train.is_open())
        cout<<"can not open "<<file_cam_a_train_name<<endl;

    file_cam_b_train.open(file_cam_b_train_name);
    if (!file_cam_b_train.is_open())
        cout<<"can not open "<<file_cam_b_train_name<<endl;


    //vectors WITH THE SAMPLES NAMES train A and B
    vector<string> samples_train_a, samples_train_b;
    samples_train_a.resize(max_id_a);
    samples_train_b.resize(max_id_b);
    while ( std::getline (file_cam_a_train,line))
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        samples_train_a[id-1]=line;
    }
    file_cam_a_train.close();
    while ( std::getline (file_cam_b_train,line))
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        samples_train_b[id-1]=line;
    }
    file_cam_b_train.close();

    //positive and negative matrixes
    vector< vector<string> > positive_train_pairs;
    positive_train_pairs.resize(min(max_id_a, max_id_b));
    for(int i=0; i<max_id_a; i++){
        string sample_a=samples_train_a[i];
        string sample_b=samples_train_b[i];
        if(!sample_a.empty() && !sample_b.empty()){
           vector<string> pair;
           pair.push_back(sample_a);
           pair.push_back(sample_b);
           positive_train_pairs[i]=pair;
        }
    }

    srand(time(NULL));
    int n_pair=0;
    while(n_pair<trainset_size){
        int randIndex_p= rand()%min(max_id_a, max_id_b);
        vector<string> pair=positive_train_pairs[randIndex_p];
        if(!pair.empty()){
             string sample_an=pair[0];
             string sample_p=pair[1];
             int randIndex_n= rand()%(max_id_b);
             string sample_n=samples_train_b[randIndex_n];
             if(!sample_n.empty() && randIndex_n!=randIndex_p){
                 file_train_an<<sample_an<<"\n";
                 file_train_p<<sample_p<<"\n";
                 file_train_n<<sample_n<<"\n";
                 n_pair++;
             }
        }
    }
    file_train_an.close();
    file_train_p.close();
    file_train_n.close();




    //val data sets----------------------------------------------------------------------------------------------------
    char file_cam_a_val_name[150], file_cam_b_val_name[150], file_an_val_name[150], file_p_val_name[150], file_n_val_name[150];

    strcpy(file_cam_a_val_name, data_folder);
    strcat(file_cam_a_val_name, "/cam_a_val.txt");
    ifstream file_cam_a_val(file_cam_a_val_name);
    if (!file_cam_a_val.is_open())
        cout<<"1. can not open "<<file_cam_a_val_name<<endl;

    strcpy(file_cam_b_val_name, data_folder);
    strcat(file_cam_b_val_name, "/cam_b_val.txt");
    ifstream file_cam_b_val(file_cam_b_val_name);
    if (!file_cam_b_val.is_open())
        cout<<"2. can not open "<<file_cam_b_val_name<<endl;

    strcpy(file_an_val_name, data_triplet_folder);
    strcat(file_an_val_name, "/val_an_fixed.txt");
    ofstream file_val_an(file_an_val_name);
    if (!file_val_an.is_open())
        cout<<"3. can not open "<<file_an_val_name<<endl;

    strcpy(file_p_val_name, data_triplet_folder);
    strcat(file_p_val_name, "/val_p_fixed.txt");
    ofstream file_val_p(file_p_val_name);
    if (!file_val_p.is_open())
        cout<<"4. can not open "<<file_p_val_name<<endl;

    strcpy(file_n_val_name, data_triplet_folder);
    strcat(file_n_val_name, "/val_n_fixed.txt");
    ofstream file_val_n(file_n_val_name);
    if (!file_val_n.is_open())
        cout<<"5. can not open "<<file_n_val_name<<endl;



    //max id in val_a
    max_id_a=0;
    while ( std::getline (file_cam_a_val,line))//
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        if(id>max_id_a)
            max_id_a=id;
    }
    file_cam_a_val.close();

    //max id in val_b
    max_id_b=0;
    while ( std::getline (file_cam_b_val,line))//
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        if(id>max_id_b)
            max_id_b=id;
    }
    file_cam_b_val.close();


    file_cam_a_val.open(file_cam_a_val_name);
    if (!file_cam_a_val.is_open())
        cout<<"can not open "<<file_cam_a_val_name<<endl;

    file_cam_b_val.open(file_cam_b_val_name);
    if (!file_cam_b_val.is_open())
        cout<<"can not open "<<file_cam_b_val_name<<endl;


    //vectors WITH THE SAMPLES NAMES val A and B
    vector<string> samples_val_a, samples_val_b;
    samples_val_a.resize(max_id_a);
    samples_val_b.resize(max_id_b);
    while ( std::getline (file_cam_a_val,line))
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        samples_val_a[id-1]=line;
    }
    file_cam_a_val.close();
    while ( std::getline (file_cam_b_val,line))
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        samples_val_b[id-1]=line;
    }
    file_cam_b_val.close();

    //positive and negative matrixes
    vector< vector<string> > positive_val_pairs;
    positive_val_pairs.resize(min(max_id_a, max_id_b));
    for(int i=0; i<max_id_a; i++){
        string sample_a=samples_val_a[i];
        string sample_b=samples_val_b[i];
        if(!sample_a.empty() && !sample_b.empty()){
           vector<string> pair;
           pair.push_back(sample_a);
           pair.push_back(sample_b);
           positive_val_pairs[i]=pair;
        }
    }

    srand(time(NULL));
    n_pair=0;
    while(n_pair<valset_size){
        int randIndex_p= rand()%min(max_id_a, max_id_b);
        vector<string> pair=positive_val_pairs[randIndex_p];
        if(!pair.empty()){
             string sample_an=pair[0];
             string sample_p=pair[1];
             int randIndex_n= rand()%(max_id_b);
             string sample_n=samples_val_b[randIndex_n];
             if(!sample_n.empty() && randIndex_n!=randIndex_p){
                 file_val_an<<sample_an<<"\n";
                 file_val_p<<sample_p<<"\n";
                 file_val_n<<sample_n<<"\n";
                 n_pair++;
             }
        }
    }
    file_val_an.close();
    file_val_p.close();
    file_val_n.close();

}


void create_triplet_data(string dataset_folder, int trainset_size, int valset_size){
    //DATA FOLDER
    char data_folder[150];
    strcpy(data_folder, dataset_folder.c_str());
    strcat(data_folder, "/DATA");
    int e=mkdir(data_folder, ACCESSPERMS);

    //DATA TRIPLET FOLDER
    char data_triplet_folder[150];
    strcpy(data_triplet_folder, data_folder);
    strcat(data_triplet_folder, "/TRIPLET");
    int etri=mkdir(data_triplet_folder, ACCESSPERMS);




    //train data sets----------------------------------------------------------------------------------------------------
    char file_cam_a_train_name[150], file_cam_b_train_name[150], file_an_train_name[150], file_p_train_name[150], file_n_train_name[150];

    strcpy(file_cam_a_train_name, data_folder);
    strcat(file_cam_a_train_name, "/cam_a_train.txt");
    ifstream file_cam_a_train(file_cam_a_train_name);
    if (!file_cam_a_train.is_open())
        cout<<"1. can not open "<<file_cam_a_train_name<<endl;

    strcpy(file_cam_b_train_name, data_folder);
    strcat(file_cam_b_train_name, "/cam_b_train.txt");
    ifstream file_cam_b_train(file_cam_b_train_name);
    if (!file_cam_b_train.is_open())
        cout<<"2. can not open "<<file_cam_b_train_name<<endl;

    strcpy(file_an_train_name, data_triplet_folder);
    strcat(file_an_train_name, "/train_an.txt");
    ofstream file_train_an(file_an_train_name);
    if (!file_train_an.is_open())
        cout<<"3. can not open "<<file_an_train_name<<endl;

    strcpy(file_p_train_name, data_triplet_folder);
    strcat(file_p_train_name, "/train_p.txt");
    ofstream file_train_p(file_p_train_name);
    if (!file_train_p.is_open())
        cout<<"4. can not open "<<file_p_train_name<<endl;

    strcpy(file_n_train_name, data_triplet_folder);
    strcat(file_n_train_name, "/train_n.txt");
    ofstream file_train_n(file_n_train_name);
    if (!file_train_n.is_open())
        cout<<"5. can not open "<<file_n_train_name<<endl;

    //max id in train_a
    string line;
    int max_id_a=0;
    while ( std::getline (file_cam_a_train,line))//
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        if(id>max_id_a)
            max_id_a=id;
    }
    file_cam_a_train.close();

    //max id in train_b
    int max_id_b=0;
    while ( std::getline (file_cam_b_train,line))//
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        if(id>max_id_b)
            max_id_b=id;
    }
    file_cam_b_train.close();


    file_cam_a_train.open(file_cam_a_train_name);
    if (!file_cam_a_train.is_open())
        cout<<"can not open "<<file_cam_a_train_name<<endl;

    file_cam_b_train.open(file_cam_b_train_name);
    if (!file_cam_b_train.is_open())
        cout<<"can not open "<<file_cam_b_train_name<<endl;


    //vectors WITH THE SAMPLES NAMES train A and B
    vector<string> samples_train_a, samples_train_b;
    samples_train_a.resize(max_id_a);
    samples_train_b.resize(max_id_b);
    while ( std::getline (file_cam_a_train,line))
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        samples_train_a[id-1]=line;
    }
    file_cam_a_train.close();
    while ( std::getline (file_cam_b_train,line))
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        samples_train_b[id-1]=line;
    }
    file_cam_b_train.close();

    //positive and negative matrixes
    vector< vector<string> > positive_train_pairs;
    positive_train_pairs.resize(min(max_id_a, max_id_b));
    for(int i=0; i<max_id_a; i++){
        string sample_a=samples_train_a[i];
        string sample_b=samples_train_b[i];
        if(!sample_a.empty() && !sample_b.empty()){
           vector<string> pair;
           pair.push_back(sample_a);
           pair.push_back(sample_b);
           positive_train_pairs[i]=pair;
        }
    }

    srand(time(NULL));
    int n_pair=0;
    while(n_pair<trainset_size){
        int randIndex_p= rand()%min(max_id_a, max_id_b);
        vector<string> pair=positive_train_pairs[randIndex_p];
        if(!pair.empty()){
             string sample_an=pair[0];
             string sample_p=pair[1];
             int randIndex_n= rand()%(max_id_b);
             string sample_n=samples_train_b[randIndex_n];
             if(!sample_n.empty() && randIndex_n!=randIndex_p){
                 file_train_an<<sample_an<<"\n";
                 file_train_p<<sample_p<<"\n";
                 file_train_n<<sample_n<<"\n";
                 n_pair++;
             }
        }
        randIndex_p= rand()%min(max_id_a, max_id_b);
        pair=positive_train_pairs[randIndex_p];
        if(!pair.empty()){
             string sample_an=pair[1];
             string sample_p=pair[0];
             int randIndex_n= rand()%(max_id_a);
             string sample_n=samples_train_a[randIndex_n];
             if(!sample_n.empty() && randIndex_n!=randIndex_p){
                 file_train_an<<sample_an<<"\n";
                 file_train_p<<sample_p<<"\n";
                 file_train_n<<sample_n<<"\n";
                 n_pair++;
             }
        }
    }
    file_train_an.close();
    file_train_p.close();
    file_train_n.close();




    //val data sets----------------------------------------------------------------------------------------------------
    char file_cam_a_val_name[150], file_cam_b_val_name[150], file_an_val_name[150], file_p_val_name[150], file_n_val_name[150];

    strcpy(file_cam_a_val_name, data_folder);
    strcat(file_cam_a_val_name, "/cam_a_val.txt");
    ifstream file_cam_a_val(file_cam_a_val_name);
    if (!file_cam_a_val.is_open())
        cout<<"1. can not open "<<file_cam_a_val_name<<endl;

    strcpy(file_cam_b_val_name, data_folder);
    strcat(file_cam_b_val_name, "/cam_b_val.txt");
    ifstream file_cam_b_val(file_cam_b_val_name);
    if (!file_cam_b_val.is_open())
        cout<<"2. can not open "<<file_cam_b_val_name<<endl;

    strcpy(file_an_val_name, data_triplet_folder);
    strcat(file_an_val_name, "/val_an.txt");
    ofstream file_val_an(file_an_val_name);
    if (!file_val_an.is_open())
        cout<<"3. can not open "<<file_an_val_name<<endl;

    strcpy(file_p_val_name, data_triplet_folder);
    strcat(file_p_val_name, "/val_p.txt");
    ofstream file_val_p(file_p_val_name);
    if (!file_val_p.is_open())
        cout<<"4. can not open "<<file_p_val_name<<endl;

    strcpy(file_n_val_name, data_triplet_folder);
    strcat(file_n_val_name, "/val_n.txt");
    ofstream file_val_n(file_n_val_name);
    if (!file_val_n.is_open())
        cout<<"5. can not open "<<file_n_val_name<<endl;

    //max id in val_a
    max_id_a=0;
    while ( std::getline (file_cam_a_val,line))//
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        if(id>max_id_a)
            max_id_a=id;
    }
    file_cam_a_val.close();

    //max id in val_b
    max_id_b=0;
    while ( std::getline (file_cam_b_val,line))//
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        if(id>max_id_b)
            max_id_b=id;
    }
    file_cam_b_val.close();


    file_cam_a_val.open(file_cam_a_val_name);
    if (!file_cam_a_val.is_open())
        cout<<"can not open "<<file_cam_a_val_name<<endl;

    file_cam_b_val.open(file_cam_b_val_name);
    if (!file_cam_b_val.is_open())
        cout<<"can not open "<<file_cam_b_val_name<<endl;


    //vectors WITH THE SAMPLES NAMES val A and B
    vector<string> samples_val_a, samples_val_b;
    samples_val_a.resize(max_id_a);
    samples_val_b.resize(max_id_b);
    while ( std::getline (file_cam_a_val,line))
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        samples_val_a[id-1]=line;
    }
    file_cam_a_val.close();
    while ( std::getline (file_cam_b_val,line))
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int id=atoi(strs[1].c_str());
        samples_val_b[id-1]=line;
    }
    file_cam_b_val.close();

    cout<<"ghoooookt"<<endl;
    //positive and negative matrixes
    vector< vector<string> > positive_val_pairs;
    positive_val_pairs.resize(max_id_a);
    for(int i=0; i<max_id_a; i++){
        string sample_a=samples_val_a[i];
        string sample_b=samples_val_b[i];
        if(!sample_a.empty() && !sample_b.empty()){
           vector<string> pair;
           pair.resize(2);
           pair[0]=sample_a;
           pair[1]=sample_b;
           positive_val_pairs[i]=pair;
        }
    }

    cout<<"ghjukt"<<endl;
    srand(time(NULL));
    n_pair=0;
    while(n_pair<valset_size){
        int randIndex_p= rand()%min(max_id_a, max_id_b);
        vector<string> pair=positive_val_pairs[randIndex_p];
        if(!pair.empty()){
             string sample_an=pair[0];
             string sample_p=pair[1];
             int randIndex_n= rand()%(max_id_b);
             string sample_n=samples_val_b[randIndex_n];
             if(!sample_n.empty() && randIndex_n!=randIndex_p){
                 file_val_an<<sample_an<<"\n";
                 file_val_p<<sample_p<<"\n";
                 file_val_n<<sample_n<<"\n";
                 n_pair++;
             }
        }
        randIndex_p= rand()%min(max_id_a, max_id_b);
        pair=positive_val_pairs[randIndex_p];
        if(!pair.empty()){
             string sample_an=pair[1];
             string sample_p=pair[0];
             int randIndex_n= rand()%(max_id_a);
             string sample_n=samples_val_a[randIndex_n];
             if(!sample_n.empty() && randIndex_n!=randIndex_p){
                 file_val_an<<sample_an<<"\n";
                 file_val_p<<sample_p<<"\n";
                 file_val_n<<sample_n<<"\n";
                 n_pair++;
             }
        }
    }
    file_val_an.close();
    file_val_p.close();
    file_val_n.close();

}


void create_test_data(string dataset_folder){
    //DATA FOLDER
    char data_folder[150];
    strcpy(data_folder, dataset_folder.c_str());
    strcat(data_folder, "/DATA");
    int e=mkdir(data_folder, ACCESSPERMS);


    //DATA TEST FOLDER
    char data_test_folder[150];
    strcpy(data_test_folder, data_folder);
    strcat(data_test_folder, "/TEST");
    int etest=mkdir(data_test_folder, ACCESSPERMS);


    char file_cam_a_test_name[150], file_cam_b_test_name[150], file_a_test_name[150], file_b_test_name[150];

    strcpy(file_cam_a_test_name, data_folder);
    strcat(file_cam_a_test_name, "/cam_a_test.txt");
    ifstream file_cam_a_test(file_cam_a_test_name);
    if (!file_cam_a_test.is_open())
        cout<<"1. can not open "<<file_cam_a_test_name<<endl;

    strcpy(file_cam_b_test_name, data_folder);
    strcat(file_cam_b_test_name, "/cam_b_test.txt");
    ifstream file_cam_b_test(file_cam_b_test_name);
    if (!file_cam_b_test.is_open())
        cout<<"2. can not open "<<file_cam_b_test_name<<endl;

    strcpy(file_a_test_name, data_test_folder);
    strcat(file_a_test_name, "/test_a.txt");
    ofstream file_test_a(file_a_test_name);
    if (!file_test_a.is_open())
        cout<<"3. can not open "<<file_a_test_name<<endl;

    strcpy(file_b_test_name, data_test_folder);
    strcat(file_b_test_name, "/test_b.txt");
    ofstream file_test_b(file_b_test_name);
    if (!file_test_b.is_open())
        cout<<"4. can not open "<<file_b_test_name<<endl;


    string linea, sample_a, sample_b, label_a, label_b;


    int st_a=0;
    while(getline (file_cam_a_test,linea))
        st_a++;
    file_cam_a_test.close();
    file_cam_a_test.open(file_cam_a_test_name);
    if (!file_cam_a_test.is_open())
        cout<<"can not open "<<file_cam_a_test_name<<endl;

    int st_b=0;
    while(getline (file_cam_b_test,linea))
        st_b++;
    file_cam_b_test.close();
    file_cam_b_test.open(file_cam_b_test_name);
    if (!file_cam_b_test.is_open())
        cout<<"can not open "<<file_cam_b_test_name<<endl;

    cout<<"st_a: "<<st_a<<endl;
    cout<<"st_b: "<<st_b<<endl;

    for(int a=0; a<st_a;a++){
        file_cam_a_test>> sample_a >> label_a;
        for(int b=0; b<st_b; b++)
            file_test_a<<sample_a<<" "<<label_a<<"\n";
    }
    for(int a=0; a<st_a;a++){

        for(int b=0; b<st_b; b++){
            file_cam_b_test>> sample_a >> label_a;
            file_test_b<<sample_a<<" "<<label_a<<"\n";
        }
        file_cam_b_test.close();
        file_cam_b_test.open(file_cam_b_test_name);
    }



    file_cam_a_test.close();
    file_cam_b_test.close();
    file_test_a.close();
    file_test_b.close();

}


