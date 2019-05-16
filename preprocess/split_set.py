import os
import json
import csv
import numpy as np
import pandas as pd
import datetime
import re
import time
import itertools
import random
import h5py

class split_set(object):
    def __init__(self,path_in):
        self._path_in=path_in

    def cross_view_split(self):
        print("cross view evaluation...")
        list_all_names = os.listdir(self._path_in)
        train_id = [1, 4, 5]
        train_list = []
        for file in list_all_names:
            cam = re.match(r"([a-z]+)([0-9]+)", file.split('_')[1], re.I)
            cam_id = int(cam[2])
            if cam_id in train_id:
                train_list.append(file)
        test_list = list(set(list_all_names) - set(train_list))
        train_fall=0
        test_fall=0
        for train in train_list:
            if 'A0' in train:
                train_fall+=1
        for test in test_list:
            if 'A0' in test:
                test_fall +=1
        print('测试集',len(test_list),'跌落数据：',test_fall)
        print('训练集',len(train_list),'跌落数据：',train_fall)
        return train_list, test_list  # Fall1_Cam1.avi_keys

    def get_skeleton_array(self,filename,num_joints=18):
        lines = open(os.path.join(self._path_in, filename), 'r').readlines()

        skeleton_set = []
        skeleton_set.append(np.zeros((100, num_joints, 2)))

        for index, line in enumerate(lines):
            data = np.asarray(list(map(float,np.array(line.strip().split())))).reshape(18,2)
            skeleton_set[0][index]=data

        # print('original shape:',len(skeleton_set),len(skeleton_set[0]),len(skeleton_set[0][0]),len(skeleton_set[0][0][0]))
        return skeleton_set

    def save_h5_file_skeleton_list(self, save_home, train_list, split='train'):
        save_name = os.path.join(save_home, 'new_file_list_' +  split + '.txt')
        with open(save_name, 'w') as fid_txt:  # fid.write(file+'\n')
            # save array list to hdf5
            save_name = os.path.join(save_home, 'new_array_list_' + split + '.h5')
            with h5py.File(save_name, 'w') as fid_h5:
                number=0
                fall_number=0
                total=0
                for fn in train_list:
                    skeleton_set = self.get_skeleton_array(fn)
                    number=number+1

                    # down sample
                    # if 'A0' in fn or number%20==0:
                    #     # print(fn)
                    #     fid_h5[fn] = skeleton_set[0][:,:, 0:2]
                    #     fid_txt.write(fn + '\n')
                    #     if 'A0' in fn:
                    #         fall_number=fall_number+1
                    #     total = total+1

                    # up sample
                    if 'A0' in fn:
                        i = 0
                        while i< 20:
                            fid_h5[fn+str(i)] = skeleton_set[0][:,:, 0:2]
                            fid_txt.write(fn+str(i) + '\n')
                            i +=1
                            fall_number +=1
                            total +=1
                    else:
                        fid_h5[fn] = skeleton_set[0][:,:, 0:2]
                        total +=1

                    # origin
                    # fid_h5[fn] = skeleton_set[0][:,:, 0:2]
                    # fid_txt.write(fn + '\n')
                    # if 'A0' in fn:
                    #     fall_number=fall_number+1
                    # total = total+1

        print(split+" fall：",fall_number)
        print(split+" total:",total)
        print("total file number",number)

if __name__ == '__main__':
    data_path = '../../samples/'
    path_out = '../../samples/'
    db = split_set(data_path)
    train_list, test_list = db.cross_view_split()
    db.save_h5_file_skeleton_list('../data/view_seq_20', train_list, split='train')
    db.save_h5_file_skeleton_list('../data/view_seq_20', test_list, split='test')
