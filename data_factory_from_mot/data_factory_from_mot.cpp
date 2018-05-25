#include "data_factory_from_mot.h"
#include <stdio.h>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <string>
#include <time.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <sstream>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;
using namespace cv;

string int2str(int a){
    stringstream ss;
    ss << a;
    string str = ss.str();
    return str;
}


void get_samples(string dataset_folder, float min_visibility){

    //VECTOR WITH THE TRAIN SEQUENCES NAMES
    char train_sequences_list_file[150];
    strcpy(train_sequences_list_file, dataset_folder.c_str());
    strcat(train_sequences_list_file, "/DATABASE/train/train_sequences_list.txt");
    ifstream train_list(train_sequences_list_file);
    if (!train_list.is_open())
          cout<<"1. can not open " << train_sequences_list_file<<endl;
    vector<string> train_sequences_list;
    string line;
    while (getline (train_list,line)) {
          train_sequences_list.push_back(line);
    }

    //SAMPLES FOLDERS
    char samples_folder[150], train_samples_folder[150];
    strcpy(samples_folder, dataset_folder.c_str());
    strcat(samples_folder, "/SAMPLES");
    strcpy(train_samples_folder, samples_folder);
    strcat(train_samples_folder, "/train");
    int e=mkdir(samples_folder, ACCESSPERMS);
    int e_tr=mkdir(train_samples_folder, ACCESSPERMS);

    //SAMPLES LIST FILE (OUTPUT)
    char train_samples_list_file_name[150];
    strcpy(train_samples_list_file_name, train_samples_folder);
    strcat(train_samples_list_file_name, "/train_samples_list.txt");
    ofstream train_samples_list_file(train_samples_list_file_name);
    if (!train_samples_list_file.is_open())
        cout<<"2. can not open "<<train_samples_list_file_name<<endl;

    int number=0;//SAMPLE NUMBER

    //FOR LOOP (FOR EVERY TRAIN DATASET)
    cout<<"Train samples extraction..."<<endl;
    for(int d=0; d<train_sequences_list.size(); d++){
        string sequence=train_sequences_list[d];
        char gt_file_name[150], frames_folder[150], ds_samples_folder[150], frame_file_name[150], sample_file_name[150];

        //SAMPLES LIST FILE (OUTPUT)
        char ds_samples_list_name[150];
        strcpy(ds_samples_list_name, train_samples_folder);
        strcat(ds_samples_list_name, "/");
        strcat(ds_samples_list_name, sequence.c_str());
        strcat(ds_samples_list_name, "_samples_list.txt");
        ofstream ds_samples_list_file(ds_samples_list_name);
        if (!ds_samples_list_file.is_open())
            cout<<"3. can not open "<<ds_samples_list_name<<endl;

        //GT FILE
        strcpy(gt_file_name, dataset_folder.c_str());
        strcat(gt_file_name, "/DATABASE/train/");
        strcat(gt_file_name, sequence.c_str());
        strcat(gt_file_name, "/gt/gt.txt");
        ifstream gt_file(gt_file_name);
        if (!gt_file.is_open())
        cout<<"4. can not open "<<gt_file_name<<endl;

        //FRAMES FOLDER
        strcpy(frames_folder, dataset_folder.c_str());
        strcat(frames_folder, "/DATABASE/train/");
        strcat(frames_folder ,sequence.c_str());
        strcat(frames_folder, "/img1/");

        //SAMPLES FOLDER (OUTPUT)
        strcpy(ds_samples_folder, train_samples_folder);
        strcat(ds_samples_folder, "/");
        strcat(ds_samples_folder, sequence.c_str());
        e=mkdir(ds_samples_folder, ACCESSPERMS);

        std::string linea;
        while ( std::getline (gt_file,linea))//   && count<50000 probe&&
        {
            number++;
            
            //GET LINE OF GT
            std::stringstream ss;
            ss.str(linea);
            std::vector<std::string> strs;
            boost::split(strs, linea, boost::is_any_of(", "));

            //LOAD FRAME
            string frame_name= "000000.jpg";
            int frame_number=atoi(strs[0].c_str());
            stringstream s1;
            s1 << frame_number;
            string fn = s1.str();
            int size1=fn.size();
            frame_name.replace(frame_name.end()-size1-4, frame_name.end()-4, fn);
            strcpy(frame_file_name, frames_folder);
            strcat(frame_file_name, frame_name.c_str());
            Mat frame=imread(frame_file_name);
            int max_width=frame.cols;
            int max_height=frame.rows;

            //ATTRIBUTES
            int id=atoi(strs[1].c_str());
            Rect roi;
            roi.x=atoi(strs[2].c_str());
            roi.y=atoi(strs[3].c_str());
            roi.width=atoi(strs[4].c_str());
            roi.height=atoi(strs[5].c_str());
            if(roi.x<0){
                roi.x=0;
            }
            if(roi.y<0){
                roi.y=0;
            }
            if((roi.x+roi.width)>max_width){
                roi.width=max_width-roi.x-1;
            }
            if((roi.y+roi.height)>max_height){
                roi.height=max_height-roi.y-1;
            }
            int conf=atoi(strs[6].c_str());
            int type=atoi(strs[7].c_str());
            float visibility=atoi(strs[8].c_str());

            //EXTRACT SAMPLE
            if (roi.width>0 && roi.height>0){
                Mat sample;
                frame(roi).copyTo(sample);
                string sample_name= "000000.png";
                stringstream s2;
                s2 << number;
                string n = s2.str();
                int size2=n.size();
                sample_name.replace(sample_name.end()-size2-4, sample_name.end()-4, n);
                strcpy(sample_file_name, ds_samples_folder);
                strcat(sample_file_name, "/");
                strcat(sample_file_name, sample_name.c_str());
                imwrite(sample_file_name, sample);

                //WRITE SAMPLES LIST
                if (conf!=0 && visibility>min_visibility && (type==1 || type==7)){
                    train_samples_list_file<<sequence<<"/"<<sample_name<<", "<<sequence<<", "<<frame_number<<", "<<id<<"\n";
                    ds_samples_list_file<<sequence<<"/"<<sample_name<<", "<<sequence<<", "<<frame_number<<", "<<id<<"\n";
                }
            }
        }
        gt_file.close();
        ds_samples_list_file.close();
    }
    train_samples_list_file.close();

    cout<<"samples extraction done"<<endl;
}


void create_pair_data(string dataset_folder, int max_t_pairs, int max_v_pairs, int ts, int oversampling, float training_identities_ratio){
    srand (time(NULL));

    //DATA FOLDER
    char data_folder[150];
    strcpy(data_folder, dataset_folder.c_str());
    strcat(data_folder, "/DATA");
    int e=mkdir(data_folder, ACCESSPERMS);

    //DATA PAIRS FOLDER
    char data_pair_folder[150];
    strcpy(data_pair_folder, data_folder);
    strcat(data_pair_folder, "/PAIR");
    int ePAIR=mkdir(data_pair_folder, ACCESSPERMS);

    //SAMPLES FOLDERS
    char samples_folder[150], train_samples_folder[150];
    strcpy(samples_folder, dataset_folder.c_str());
    strcat(samples_folder, "/SAMPLES");
    strcpy(train_samples_folder, samples_folder);
    strcat(train_samples_folder, "/train");

    //SAMPLES LIST FILE
    char samples_list_file_name[150];
    strcpy(samples_list_file_name, train_samples_folder);
    strcat(samples_list_file_name, "/train_samples_list.txt");
    ifstream samples_list_file(samples_list_file_name);
    if (!samples_list_file.is_open())
        cout<<"1. can not open "<<samples_list_file_name<<endl;

    char samples_list_file_name_new[150];
    strcpy(samples_list_file_name_new, data_folder);
    strcat(samples_list_file_name_new, "/train_samples_list.txt");
    ofstream samples_list_file_new(samples_list_file_name_new);
    if (!samples_list_file_new.is_open())
        cout<<"2. can not open "<<samples_list_file_name_new<<endl;

    string line;
    int max_id=0;
    int previous_id=0;
    int new_id=0;
    int max_frame=0;
    while ( std::getline (samples_list_file,line))//
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(","));
        int frame=atoi(strs[2].c_str());
        int id=atoi(strs[3].c_str());
        if (id!=previous_id){
            previous_id=id;
            new_id++;
        }
        samples_list_file_new<<strs[0]<<","<<strs[1]<<","<<strs[2]<<", "<<new_id<<"\n";
        if(new_id>max_id)
            max_id=new_id;
        if (frame>max_frame)
            max_frame=frame;
    }
    samples_list_file.close();
    samples_list_file_new.close();

    if(max_id>1){
        samples_list_file.open(samples_list_file_name_new);

        //MATRIX WITH THE SAMPLES NAMES
        vector<vector<string> > samples_matrix;
        int height = max_id;
        int width = max_frame;
        samples_matrix.resize(height);
        for(int i = 0; i < height; i++) samples_matrix[i].resize(width);
        while ( std::getline (samples_list_file,line))
        {
            std::stringstream ss;
            ss.str(line);
            std::vector<std::string> strs;
            boost::split(strs, line, boost::is_any_of(","));
            int frame=atoi(strs[2].c_str());
            int id=atoi(strs[3].c_str());
            string label_a="0000";
            stringstream s3;
            s3 << id;
            string idn = s3.str();
            int size=idn.size();
            label_a.replace(label_a.end()-size, label_a.end(), idn);
            char sample[100];
            strcpy(sample, strs[0].c_str());
            strcat(sample, " ");
            strcat(sample, label_a.c_str());
            vector<string> row=samples_matrix[id-1];
            row[frame-1]=sample;
            samples_matrix[id-1]=row;
        }

        //TRAIN-VAL IDENTITIES DIVISION
        cout<<"Train-val identities division..."<<endl;
        vector<vector<string> > val_samples_matrix, train_samples_matrix;
        int train_size=0;
        int val_size=0;
        for(int r=0; r<samples_matrix.size(); r=r+oversampling){
            vector<string> identity=samples_matrix[r];
            if(!identity.empty()){
                int id=r;
                int grupos=max_id/100;
                int rest=max_id%100;
                if(id<grupos*100){
                    if((id%100)< cvRound(training_identities_ratio*100.0)){//train
                        train_samples_matrix.push_back(identity);
                        train_size++;
                    }else{//val
                        val_samples_matrix.push_back(identity);
                        val_size++;
                    }
                }else{
                    if(float(float(id)/float(rest)) <= training_identities_ratio){//train
                        train_samples_matrix.push_back(identity);
                        train_size++;
                    }else{//val
                        val_samples_matrix.push_back(identity);
                        val_size++;
                    }
                }
            }
        }

        if(train_size>1){
            //TRAIN_A (OUTPUT)
            char train_a_file_name[150];
            strcpy(train_a_file_name, data_pair_folder);
            strcat(train_a_file_name, "/train_a.txt");
            ofstream train_a_file(train_a_file_name);
            if (!train_a_file.is_open())
                cout<<"can not open train_a.txt"<<endl;

            //TRAIN_B (OUTPUT)
            char train_b_file_name[150];
            strcpy(train_b_file_name, data_pair_folder);
            strcat(train_b_file_name, "/train_b.txt");
            ofstream train_b_file(train_b_file_name);
            if (!train_b_file.is_open())
                cout<<"can not open train_b.txt"<<endl;

            //TRAIN PAIRS CREATION
            cout<<"Training pairs creation..."<<endl;
            vector<vector<string> > train_pairs;
            int t_pairs=0;
            while(t_pairs<max_t_pairs){
                for(int r=0; r<train_samples_matrix.size(); r++){
                    vector<string> row=train_samples_matrix[r];//identity
                    for(int c=ts; c<row.size(); c=c+oversampling){
                        for(int d=0; d<ts; d=d+1){
                            if(t_pairs<max_t_pairs){
                                if(!train_samples_matrix[r][c].empty() && !train_samples_matrix[r][c-d].empty()){//positive pair
                                    string a_sample=train_samples_matrix[r][c];
                                    string bp_sample=train_samples_matrix[r][c-d];
                                    //negative pair search
                                    string bn_sample;
                                    bool impostor_found=false;
                                    int n_searches=0;
                                    while(impostor_found==false && n_searches<1000){
                                        int x=rand() % train_samples_matrix.size();
                                        int f=rand() % train_samples_matrix[x].size();
                                        if(!train_samples_matrix[x][f].empty() && x!=r){
                                            bn_sample=train_samples_matrix[x][f];
                                            impostor_found=true;
                                        }
                                        n_searches++;
                                    }
                                    if(impostor_found){//add pairs
                                        vector<string> pair_p;
                                        pair_p.push_back(a_sample);
                                        pair_p.push_back(bp_sample);
                                        train_pairs.push_back(pair_p);
                                        t_pairs++;
                                        vector<string> pair_n;
                                        pair_n.push_back(a_sample);
                                        pair_n.push_back(bn_sample);
                                        train_pairs.push_back(pair_n);
                                        t_pairs++;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            //random generation for train
            int n_tracklets=train_pairs.size();
            vector<int> random_index;
            for(int t=0; t<n_tracklets; t++)
                random_index.push_back(t);
            std::random_shuffle ( random_index.begin(), random_index.end() );
            for(int t=0; t<n_tracklets; t++){
                //an, p, n
                train_a_file << train_pairs[random_index[t]][0]<<"\n";
                train_b_file << train_pairs[random_index[t]][1]<<"\n";
            }

            //CLOSE OUTPUT FILES
            train_a_file.close();
            train_b_file.close();

        }else{
            cout<<"     Not enough identities to create train data"<<endl;
        }

        if(val_size>1){

            //VAL_A (OUTPUT)
            char val_a_file_name[150];
            strcpy(val_a_file_name, data_pair_folder);
            strcat(val_a_file_name, "/val_a.txt");
            ofstream val_a_file(val_a_file_name);
            if (!val_a_file.is_open())
                cout<<"can not open val_a.txt"<<endl;

            //VAL_B (OUTPUT)
            char val_b_file_name[150];
            strcpy(val_b_file_name, data_pair_folder);
            strcat(val_b_file_name, "/val_b.txt");
            ofstream val_b_file(val_b_file_name);
            if (!val_b_file.is_open())
                cout<<"can not open val_b.txt"<<endl;

            //VAL PAIRS CREATION
            cout<<"Validation pairs creation..."<<endl;
            vector<vector<string> > val_pairs;
            int v_pairs=0;
            while(v_pairs<max_v_pairs){
                for(int r=0; r<val_samples_matrix.size(); r++){
                    vector<string> row=val_samples_matrix[r];//identity
                    for(int c=ts; c<row.size(); c=c+oversampling){
                        for(int d=0; d<ts; d=d+1){
                            if(v_pairs<max_v_pairs){
                                if(!val_samples_matrix[r][c].empty() && !val_samples_matrix[r][c-d].empty()){//positive pair
                                    string a_sample=val_samples_matrix[r][c];
                                    string bp_sample=val_samples_matrix[r][c-d];
                                    //negative pair search
                                    string bn_sample;
                                    bool impostor_found=false;
                                    int n_searches=0;
                                    while(impostor_found==false && n_searches<1000){
                                        int x=rand() % val_samples_matrix.size();
                                        int f=rand() % val_samples_matrix[x].size();
                                        if(!val_samples_matrix[x][f].empty() && x!=r){
                                            bn_sample=val_samples_matrix[x][f];
                                            impostor_found=true;
                                        }
                                        n_searches++;
                                    }
                                    if(impostor_found){//add pairs
                                        vector<string> pair_p;
                                        pair_p.push_back(a_sample);
                                        pair_p.push_back(bp_sample);
                                        val_pairs.push_back(pair_p);
                                        v_pairs++;
                                        vector<string> pair_n;
                                        pair_n.push_back(a_sample);
                                        pair_n.push_back(bn_sample);
                                        val_pairs.push_back(pair_n);
                                        v_pairs++;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            //random generation for train
            int n_tracklets=val_pairs.size();
            vector<int> random_index;
            for(int t=0; t<n_tracklets; t++)
                random_index.push_back(t);
            std::random_shuffle ( random_index.begin(), random_index.end() );
            for(int t=0; t<n_tracklets; t++){
                //an, p, n
                val_a_file << val_pairs[random_index[t]][0]<<"\n";
                val_b_file << val_pairs[random_index[t]][1]<<"\n";
            }

            //CLOSE OUTPUT FILES
            val_a_file.close();
            val_b_file.close();

        }else{
            cout<<"     Not enough identities to create val data"<<endl;
        }

    }else{
        cout<<"  Not enough identities to create data"<<endl;
    }
}

void create_triplet_data(string dataset_folder, int max_t_triplets, int max_v_triplets, int ts, int oversampling, float training_identities_ratio){
    srand (time(NULL));

    //DATA FOLDER
    char data_folder[150];
    strcpy(data_folder, dataset_folder.c_str());
    strcat(data_folder, "/DATA");
    int e=mkdir(data_folder, ACCESSPERMS);

    //DATA TRIPLETS FOLDER
    char data_triplet_folder[150];
    strcpy(data_triplet_folder, data_folder);
    strcat(data_triplet_folder, "/TRIPLETS");
    int et=mkdir(data_triplet_folder, ACCESSPERMS);

    //SAMPLES FOLDERS
    char samples_folder[150], train_samples_folder[150];
    strcpy(samples_folder, dataset_folder.c_str());
    strcat(samples_folder, "/SAMPLES");
    strcpy(train_samples_folder, samples_folder);
    strcat(train_samples_folder, "/train");

    //SAMPLES LIST FILE
    char samples_list_file_name[150];
    strcpy(samples_list_file_name, train_samples_folder);
    strcat(samples_list_file_name, "/train_samples_list.txt");
    ifstream samples_list_file(samples_list_file_name);
    if (!samples_list_file.is_open())
        cout<<"can not open "<<samples_list_file_name<<endl;

    char samples_list_file_name_new[150];
    strcpy(samples_list_file_name_new, data_folder);
    strcat(samples_list_file_name_new, "/train_samples_list.txt");
    ofstream samples_list_file_new(samples_list_file_name_new);
    if (!samples_list_file_new.is_open())
        cout<<"can not open "<<samples_list_file_name_new<<endl;

    string line;
    int max_id=0;
    int previous_id=0;
    int new_id=0;
    int max_frame=0;
    while ( std::getline (samples_list_file,line))//
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(","));
        int frame=atoi(strs[2].c_str());
        int id=atoi(strs[3].c_str());
        if (id!=previous_id){
            previous_id=id;
            new_id++;
        }
        samples_list_file_new<<strs[0]<<","<<strs[1]<<","<<strs[2]<<", "<<new_id<<"\n";
        if(new_id>max_id)
            max_id=new_id;
        if (frame>max_frame)
            max_frame=frame;
    }
    samples_list_file.close();
    samples_list_file_new.close();

    if(max_id>1){
        samples_list_file.open(samples_list_file_name_new);

        //MATRIX WITH THE SAMPLES NAMES
        vector<vector<string> > samples_matrix;
        int height = max_id;
        int width = max_frame;
        samples_matrix.resize(height);
        for(int i = 0; i < height; i++) samples_matrix[i].resize(width);
        while ( std::getline (samples_list_file,line))
        {
            std::stringstream ss;
            ss.str(line);
            std::vector<std::string> strs;
            boost::split(strs, line, boost::is_any_of(","));
            int frame=atoi(strs[2].c_str());
            int id=atoi(strs[3].c_str());
            string label_a="0000";
            stringstream s3;
            s3 << id;
            string idn = s3.str();
            int size=idn.size();
            label_a.replace(label_a.end()-size, label_a.end(), idn);
            char sample[100];
            strcpy(sample, strs[0].c_str());
            strcat(sample, " ");
            strcat(sample, label_a.c_str());
            vector<string> row=samples_matrix[id-1];
            row[frame-1]=sample;
            samples_matrix[id-1]=row;
        }

        //TRAIN-VAL IDENTITIES DIVISION----------------------------------------------
        cout<<"Train-val identities division..."<<endl;
        vector<vector<string> > val_samples_matrix, train_samples_matrix;
        int train_size=0;
        int val_size=0;
        for(int r=0; r<samples_matrix.size(); r=r+oversampling){
            vector<string> identity=samples_matrix[r];
            if(!identity.empty()){
                int id=r;
                int grupos=max_id/100;
                int rest=max_id%100;
                if(id<grupos*100){
                    if((id%100)< cvRound(training_identities_ratio*100.0)){//train
                        train_samples_matrix.push_back(identity);
                        train_size++;
                    }else{//val
                        val_samples_matrix.push_back(identity);
                        val_size++;
                    }
                }else{
                    if(float(float(id)/float(rest)) <= training_identities_ratio){//train
                        train_samples_matrix.push_back(identity);
                        train_size++;
                    }else{//val
                        val_samples_matrix.push_back(identity);
                        val_size++;
                    }
                }
            }
        }

        if(train_size>1){

            //TRAIN_AN (OUTPUT)
            char train_an_file_name[150];
            strcpy(train_an_file_name, data_triplet_folder);
            strcat(train_an_file_name, "/train_an.txt");
            ofstream train_an_file(train_an_file_name);
            if (!train_an_file.is_open())
                cout<<"can not open train_an.txt"<<endl;

            //TRAIN_P (OUTPUT)
            char train_p_file_name[150];
            strcpy(train_p_file_name, data_triplet_folder);
            strcat(train_p_file_name, "/train_p.txt");
            ofstream train_p_file(train_p_file_name);
            if (!train_p_file.is_open())
                cout<<"can not open train_p.txt"<<endl;

            //TRAIN_N (OUTPUT)
            char train_n_file_name[150];
            strcpy(train_n_file_name, data_triplet_folder);
            strcat(train_n_file_name, "/train_n.txt");
            ofstream train_n_file(train_n_file_name);
            if (!train_n_file.is_open())
                cout<<"can not open train_n.txt"<<endl;

            //TRAIN TRIPLETS CREATION
            cout<<"Training triplets creation..."<<endl;
            vector<vector<string> > train_triplets;
            int t_triplets=0;
            while(t_triplets<max_t_triplets){
                for(int r=0; r<train_samples_matrix.size(); r++){
                    vector<string> row=train_samples_matrix[r];//identity
                    for(int c=ts; c<row.size(); c=c+oversampling){
                        for(int d=0; d<ts; d=d+1){
                            if(t_triplets<max_t_triplets){
                                if(!train_samples_matrix[r][c].empty() && !train_samples_matrix[r][c-d].empty()){//positive pair
                                    string an_sample=train_samples_matrix[r][c];
                                    string p_sample=train_samples_matrix[r][c-d];
                                    //negative pair search
                                    string n_sample;
                                    bool impostor_found=false;
                                    int n_searches=0;
                                    while(impostor_found==false && n_searches<1000){
                                        int x=rand() % train_samples_matrix.size();
                                        int f=rand() % train_samples_matrix[x].size();
                                        if(!train_samples_matrix[x][f].empty() && x!=r){
                                            n_sample=train_samples_matrix[x][f];
                                            impostor_found=true;
                                        }
                                        n_searches++;
                                    }
                                    if(impostor_found){//add triplet
                                        vector<string> triplet;
                                        triplet.push_back(an_sample);
                                        triplet.push_back(p_sample);
                                        triplet.push_back(n_sample);
                                        train_triplets.push_back(triplet);
                                        t_triplets++;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            //random generation for train
            int n_tracklets=train_triplets.size();
            vector<int> random_index;
            for(int t=0; t<n_tracklets; t++)
                random_index.push_back(t);
            std::random_shuffle ( random_index.begin(), random_index.end() );
            for(int t=0; t<n_tracklets; t++){
                //an, p, n
                train_an_file << train_triplets[random_index[t]][0]<<"\n";
                train_p_file << train_triplets[random_index[t]][1]<<"\n";
                train_n_file << train_triplets[random_index[t]][2]<<"\n";
            }

            //CLOSE OUTPUT FILES
            train_an_file.close();
            train_p_file.close();
            train_n_file.close();

        }else{
            cout<<"     Not enough identities to create train data"<<endl;
        }

        if(val_size>1){
            //VAL_AN (OUTPUT)
            char val_an_file_name[150];
            strcpy(val_an_file_name, data_triplet_folder);
            strcat(val_an_file_name, "/val_an.txt");
            ofstream val_an_file(val_an_file_name);
            if (!val_an_file.is_open())
                cout<<"can not open val_an.txt"<<endl;

            //VAL_P (OUTPUT)
            char val_p_file_name[150];
            strcpy(val_p_file_name, data_triplet_folder);
            strcat(val_p_file_name, "/val_p.txt");
            ofstream val_p_file(val_p_file_name);
            if (!val_p_file.is_open())
                cout<<"can not open val_p.txt"<<endl;

            //VAL_N (OUTPUT)
            char val_n_file_name[150];
            strcpy(val_n_file_name, data_triplet_folder);
            strcat(val_n_file_name, "/val_n.txt");
            ofstream val_n_file(val_n_file_name);
            if (!val_n_file.is_open())
                cout<<"can not open val_n.txt"<<endl;

            //VAL TRIPLETS CREATION
            cout<<"Validation triplets creation..."<<endl;
            vector<vector<string> > val_triplets;
            int v_triplets=0;
            while(v_triplets<max_v_triplets){
                for(int r=0; r<val_samples_matrix.size(); r++){
                    vector<string> row=val_samples_matrix[r];//identity
                    for(int c=ts; c<row.size(); c=c+oversampling){
                        for(int d=0; d<ts; d=d+1){
                            if(v_triplets<max_v_triplets){
                                if(!val_samples_matrix[r][c].empty() && !val_samples_matrix[r][c-d].empty()){//tengo par positivo
                                    string an_sample=val_samples_matrix[r][c];
                                    string p_sample=val_samples_matrix[r][c-d];
                                    //negative pair search
                                    string n_sample;
                                    bool impostor_found=false;
                                    int n_searches=0;
                                    while(impostor_found==false && n_searches<1000){
                                        int x=rand() % val_samples_matrix.size();
                                        int f=rand() % val_samples_matrix[x].size();
                                        if(!val_samples_matrix[x][f].empty() && x!=r){
                                            n_sample=val_samples_matrix[x][f];
                                            impostor_found=true;
                                        }
                                        n_searches++;
                                    }
                                    if(impostor_found){//add triplet
                                        vector<string> triplet;
                                        triplet.push_back(an_sample);
                                        triplet.push_back(p_sample);
                                        triplet.push_back(n_sample);
                                        val_triplets.push_back(triplet);
                                        v_triplets++;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            //random generation for train
            int n_tracklets=val_triplets.size();
            vector<int> random_index;
            for(int t=0; t<n_tracklets; t++)
                random_index.push_back(t);
            std::random_shuffle ( random_index.begin(), random_index.end() );
            for(int t=0; t<n_tracklets; t++){
                //an, p, n
                val_an_file << val_triplets[random_index[t]][0]<<"\n";
                val_p_file << val_triplets[random_index[t]][1]<<"\n";
                val_n_file << val_triplets[random_index[t]][2]<<"\n";
            }

            //CLOSE OUTPUT FILES
            val_an_file.close();
            val_p_file.close();
            val_n_file.close();

        }else{
            cout<<"     Not enough identities to create val data"<<endl;
        }

    }else{
        cout<<"  Not enough identities to create data"<<endl;
    }
}

void create_tracklet_data(string dataset_folder, int max_t_tracklets, int max_v_tracklets, int ts, int td, int oversampling, float training_identities_ratio){
    srand (time(NULL));

    //DATA FOLDER
    char data_folder[150];
    strcpy(data_folder, dataset_folder.c_str());
    strcat(data_folder, "/DATA");
    int e=mkdir(data_folder, ACCESSPERMS);

    //DATA PAIRS FOLDER
    char data_tracklet_folder[150];
    strcpy(data_tracklet_folder, data_folder);
    strcat(data_tracklet_folder, "/TRACKLET");
    int eT=mkdir(data_tracklet_folder, ACCESSPERMS);

    //SAMPLES FOLDERS
    char samples_folder[150], train_samples_folder[150];
    strcpy(samples_folder, dataset_folder.c_str());
    strcat(samples_folder, "/SAMPLES");
    strcpy(train_samples_folder, samples_folder);
    strcat(train_samples_folder, "/train");

    //SAMPLES LIST FILE
    char samples_list_file_name[150];
    strcpy(samples_list_file_name, train_samples_folder);
    strcat(samples_list_file_name, "/train_samples_list.txt");
    ifstream samples_list_file(samples_list_file_name);
    if (!samples_list_file.is_open())
        cout<<"can not open "<<samples_list_file_name<<endl;

    char samples_list_file_name_new[150];
    strcpy(samples_list_file_name_new, data_folder);
    strcat(samples_list_file_name_new, "/train_samples_list.txt");
    ofstream samples_list_file_new(samples_list_file_name_new);
    if (!samples_list_file_new.is_open())
        cout<<"can not open "<<samples_list_file_name_new<<endl;

    string line;
    int max_id=0;
    int previous_id=0;
    int new_id=0;
    int max_frame=0;
    while ( std::getline (samples_list_file,line))//
    {
        std::stringstream ss;
        ss.str(line);
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(","));
        int frame=atoi(strs[2].c_str());
        int id=atoi(strs[3].c_str());
        if (id!=previous_id){
            previous_id=id;
            new_id++;
        }
        samples_list_file_new<<strs[0]<<","<<strs[1]<<","<<strs[2]<<", "<<new_id<<"\n";
        if(new_id>max_id)
            max_id=new_id;
        if (frame>max_frame)
            max_frame=frame;
    }
    samples_list_file.close();
    samples_list_file_new.close();

    if(max_id>1){
        samples_list_file.open(samples_list_file_name_new);

        //MATRIX WITH THE SAMPLES NAMES
        vector<vector<string> > samples_matrix;
        int height = max_id;
        int width = max_frame;
        samples_matrix.resize(height);
        for(int i = 0; i < height; i++) samples_matrix[i].resize(width);
        while ( std::getline (samples_list_file,line))
        {
            std::stringstream ss;
            ss.str(line);
            std::vector<std::string> strs;
            boost::split(strs, line, boost::is_any_of(","));
            int frame=atoi(strs[2].c_str());
            int id=atoi(strs[3].c_str());
            string label_a="0000";
            stringstream s3;
            s3 << id;
            string idn = s3.str();
            int size=idn.size();
            label_a.replace(label_a.end()-size, label_a.end(), idn);
            char sample[100];
            strcpy(sample, strs[0].c_str());
            strcat(sample, " ");
            strcat(sample, label_a.c_str());
            vector<string> row=samples_matrix[id-1];
            row[frame-1]=sample;
            samples_matrix[id-1]=row;
        }

        //TRAIN-VAL IDENTITIES DIVISION----------------------------------------------
        cout<<"Train-val identities division..."<<endl;
        vector<vector<string> > val_samples_matrix, train_samples_matrix;
        int train_size=0;
        int val_size=0;
        for(int r=0; r<samples_matrix.size(); r=r+oversampling){
            vector<string> identity=samples_matrix[r];
            if(!identity.empty()){
                int id=r;
                int grupos=max_id/100;
                int rest=max_id%100;
                if(id<grupos*100){
                    if((id%100)< cvRound(training_identities_ratio*100.0)){//train
                        train_samples_matrix.push_back(identity);
                        train_size++;
                    }else{//val
                        val_samples_matrix.push_back(identity);
                        val_size++;
                    }
                }else{
                    if(float(float(id)/float(rest)) <= training_identities_ratio){//train
                        train_samples_matrix.push_back(identity);
                        train_size++;
                    }else{//val
                        val_samples_matrix.push_back(identity);
                        val_size++;
                    }
                }
            }
        }

        if(train_size>1){
            //OPEN OUTPUT FILES
            vector<shared_ptr<ofstream> > files;
            for(int d=0; d<td; d++){
                char file_name[150];
                strcpy(file_name, data_tracklet_folder);
                strcat(file_name, "/train_");
                strcat(file_name, int2str(d).c_str());
                strcat(file_name, ".txt");
                files.push_back( make_shared<ofstream>( file_name ) );
            }

            //TRAIN TRACKLETS CREATION
            cout<<"Training tracklets creation..."<<endl;
            vector<vector<string>> tracklets_matrix;//tracklets_matrix, height=number of tracklets, width=tracklet depth
            int t_tracklets=0;
            while(t_tracklets<max_t_tracklets){
                for(int r=0; r<train_samples_matrix.size(); r++){
                    vector<string> row=train_samples_matrix[r];//identity
                    for(int c=td; c<row.size(); c=c+oversampling){
                        bool condition=true;
                        //check id the identity have enough contiguous frames representations to create a tracklet
                        for(int d=1; d<td; d++){
                            if(train_samples_matrix[r][c-d].empty())
                                condition=false;
                        }

                        if(condition){//tracklet creation
                            for(int n=0; n<ts; n++){
                                if(t_tracklets<max_t_tracklets){

                                    vector<string> positive_tracklet, negative_tracklet;
                                    positive_tracklet.resize(td);
                                    for(int d=1; d<td; d++){
                                        positive_tracklet[d]=train_samples_matrix[r][c-d];
                                    }
                                    //search the positive
                                    bool positive_found=false;
                                    if(c+n<row.size()){
                                        if(!train_samples_matrix[r][c+n].empty())
                                        {
                                            positive_tracklet[0]=train_samples_matrix[r][c+n];
                                            positive_found=true;
                                        }
                                    }
                                    if(positive_found){
                                        //search of the impostor
                                        string impostor;
                                        bool impostor_found=false;
                                        int n_searches=0;
                                        while(impostor_found==false && n_searches<1000){
                                            int x=rand() % train_samples_matrix.size();
                                            int f=rand() % row.size();
                                            if(!train_samples_matrix[x][f].empty() && x!=r){
                                                impostor=train_samples_matrix[x][f];
                                                impostor_found=true;
                                            }
                                        }
                                        if(impostor_found){
                                            negative_tracklet=positive_tracklet;
                                            negative_tracklet[0]=impostor;
                                            tracklets_matrix.push_back(positive_tracklet);
                                            t_tracklets++;
                                            tracklets_matrix.push_back(negative_tracklet);
                                            t_tracklets++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            //random generation for train
            int n_tracklets=tracklets_matrix.size();
            vector<int> random_index;
            for(int t=0; t<n_tracklets; t++)
                random_index.push_back(t);
            std::random_shuffle ( random_index.begin(), random_index.end() );
            for(int t=0; t<n_tracklets; t++){
                for(int d=0; d<td; d++)
                    *files[d]<<tracklets_matrix[random_index[t]][d]<<"\n";
            }
            //close OUTPUT FILES
            for(int d=0; d<td; d++)
                (*files[d]).close();

        }else{
            cout<<"     Not enough identities to create train data"<<endl;
        }

        if(val_size>1){
            //OPEN OUTPUT FILES
            vector<shared_ptr<ofstream>> files;
            for(int d=0; d<td; d++){
                char file_name[150];
                strcpy(file_name, data_tracklet_folder);
                strcat(file_name, "/val_");
                strcat(file_name, int2str(d).c_str());
                strcat(file_name, ".txt");
                files.push_back( make_shared<ofstream>( file_name ) );
            }

            //TRAIN TRACKLETS CREATION
            cout<<"Validation tracklets creation..."<<endl;
            vector<vector<string>> tracklets_matrix;//tracklets_matrix, height=number of tracklets, width=tracklet depth
            int v_tracklets=0;
            while(v_tracklets<max_v_tracklets){
                for(int r=0; r<val_samples_matrix.size(); r++){
                    vector<string> row=val_samples_matrix[r];//identity
                    for(int c=td; c<row.size(); c=c+oversampling){
                        bool condition=true;

                        //check id the identity have enough contiguous frames representations to create a tracklet
                        for(int d=1; d<td; d++){
                                if(val_samples_matrix[r][c-d].empty())
                                    condition=false;
                        }

                        if(condition){//tracklet creation
                            for(int n=0; n<ts; n++){
                                if(v_tracklets<max_v_tracklets){
                                    vector<string> positive_tracklet, negative_tracklet;
                                    positive_tracklet.resize(td);
                                    for(int d=1; d<td; d++){
                                        positive_tracklet[d]=val_samples_matrix[r][c-d];
                                    }
                                    //search the positive
                                    bool positive_found=false;
                                    if(c+n<row.size()){
                                        if(!val_samples_matrix[r][c+n].empty())
                                        {
                                            positive_tracklet[0]=val_samples_matrix[r][c+n];
                                            positive_found=true;
                                        }
                                    }
                                    if(positive_found){
                                        //search of the impostor
                                        string impostor;
                                        bool impostor_found=false;
                                        int n_searches=0;
                                        while(impostor_found==false && n_searches<1000){
                                            int x=rand() % val_samples_matrix.size();
                                            int f=rand() % row.size();
                                            if(!val_samples_matrix[x][f].empty() && x!=r){
                                                impostor=val_samples_matrix[x][f];
                                                impostor_found=true;
                                            }
                                        }
                                        if(impostor_found){
                                            negative_tracklet=positive_tracklet;
                                            negative_tracklet[0]=impostor;
                                            tracklets_matrix.push_back(positive_tracklet);
                                            v_tracklets++;
                                            tracklets_matrix.push_back(negative_tracklet);
                                            v_tracklets++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            //random generation for val
            int n_tracklets=tracklets_matrix.size();
            vector<int> random_index;
            for(int t=0; t<n_tracklets; t++)
                random_index.push_back(t);
            std::random_shuffle ( random_index.begin(), random_index.end() );
            for(int t=0; t<n_tracklets; t++){
                for(int d=0; d<td; d++)
                    *files[d]<<tracklets_matrix[random_index[t]][d]<<"\n";
            }
            //close OUTPUT FILES
            for(int d=0; d<td; d++)
                (*files[d]).close();

        }else{
            cout<<"     Not enough identities to create val data"<<endl;
        }

    }else{
        cout<<"  Not enough identities to create data"<<endl;
    }

}
