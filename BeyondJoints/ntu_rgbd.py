# !/usr/bin/env python
# -*- coding:utf-8 -*-
import os
os.environ['KERAS_BACKEND'] = 'theano'
import numpy as np
import random
import h5py
# import cv2

class ntu_rgbd(object):
    def __init__(self, data_path):
        self._data_path = data_path

    def skeleton_miss_list(self):
        lines = open('data/samples_with_missing_skeletons.txt', encoding='utf-8', mode = 'r').readlines()
        return [line.strip()+'.skeleton' for line in lines]

    # def get_multi_subject_list(self):
    #     lines = open('data/samples_with_multi_subjects.txt', 'r').readlines()
    #     return [line.strip() for line in lines]

    def filter_list(self, file_list):
        miss_list = self.skeleton_miss_list()
        return list(set(file_list)-set(miss_list))

    def check_list_by_frame_num(self):
        all_list = os.listdir(self._data_path)
        all_list = self.filter_list(all_list)
        for filename in all_list:
            lines = open(os.path.join(self._data_path, filename), 'r').readlines()
            step1 = int(lines[0].strip())
            step2 = lines.count('25\r\n')
            if step2 != step1 and step2 != 2*step1 and step2 != 3*step1:
                print(filename, step1, step2)

    def cross_subject_split(self):
        print ('cross subject evaluation ...')
        #ID of training subjects P 1-40 /20 for train, 20 for test
        trn_sub = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
        all_list = os.listdir(self._data_path)
        trn_list = [file for file in all_list if int(file[9:12]) in trn_sub]
        tst_list = list(set(all_list) - set(trn_list))
        # filter file list with missing skeleton
        trn_list = self.filter_list(trn_list)
        tst_list = self.filter_list(tst_list)
        return trn_list, tst_list
        
    def get_all_name(self):
        all_list = os.listdir(self._data_path)
        all_list = self.filter_list(all_list)
        return all_list
        
    def get_test_name(self):
        self._data_path = 'test/AllSkeletonFiles_remove_nan_nolabel/'
        all_list = os.listdir(self._data_path)
        return all_list
        
    def cross_view_split(self):
        print('cross view evaluation ...')
        # Cxxx
        trn_view = [2, 3]
        all_list = os.listdir(self._data_path)
        trn_list = [file for file in all_list if int(file[5:8]) in trn_view]
        tst_list = list(set(all_list) - set(trn_list))
        # filter file list with missing skeleton
        trn_list = self.filter_list(trn_list)
        tst_list = self.filter_list(tst_list)
        return trn_list, tst_list
        
    def save_h5_file_skeleton_list(self, save_home, trn_list, split='train', angle=False):
        # save file list to txt
        save_name = os.path.join(save_home, 'new_file_list_' +  split + '.txt')
        with open(save_name, 'w') as fid_txt:  # fid.write(file+'\n')
            # save array list to hdf5
            save_name = os.path.join(save_home, 'new_array_list_' + split + '.h5')
            with h5py.File(save_name, 'w') as fid_h5:
                number=0
                fall_number=0
                total=0
                for fn in trn_list:
                    skeleton_set, pid_set, std_set = self.person_position_std(fn)
                    # filter skeleton by standard value
                    count = 0
                    number=number+1
                    #if 'A043' in fn or number%30==0:
                    for idx2 in range(len(pid_set)):
                        if (std_set[idx2][0] > 0.002 or std_set[idx2][1] > 0.002) and (std_set[idx2][0]!=0 and std_set[idx2][1]!=0):
                            count = count + 1
                            name=fn+pid_set[idx2]
                            if angle:
                                fid_h5[name] = skeleton_set[idx2][:,:, 3:]
                            else:
                                fid_h5[name] = skeleton_set[idx2][:,:, 0:3]
                                total=total+1
                            fid_txt.write(name + '\n')
                            if 'A043' in name:
                                fall_number=fall_number+1
                            total=total+1
                    if count == 0:
                        std_sum = [np.sum(it) for it in std_set]
                        idx2 = np.argmax(std_sum)
                        name=fn+pid_set[idx2]
                        if angle:
                            fid_h5[name] = skeleton_set[idx2][:,:, 3:]
                        else:
                            fid_h5[name] = skeleton_set[idx2][:,:, 0:3]
                        fid_txt.write(name + '\n')
                        if 'A043' in name:
                            fall_number=fall_number+1
                        total=total+1

        print(split+" fall：",fall_number)
        print(split+" total:",total)
        print("total file number",number)

    def person_position_std(self, filename, num_joints=25):
        lines = open(os.path.join(self._data_path, filename), 'r').readlines()
        step = int(lines[0].strip())
        pid_set = []  #72057594037931368 文件里面的这串数字，不知道什么作用
        # idx_set length of sequence
        idx_set = []
        skeleton_set = []
        start = 1
        sidx = [0,1,2,7,8,9,10]
        while start < len(lines): # and idx < step
            if lines[start].strip()=='25':
                pid = lines[start-1].split()[0]
                # print('pid: '+str(pid))
                # for x in lines[start+1:start+26]:
                #     print(np.asarray(list(map(float,np.array(x.strip().split())[sidx]))).shape)

                if pid not in pid_set:
                    idx_set.append(0)
                    pid_set.append(pid)
                    skeleton_set.append(np.zeros((step, num_joints, 7)))
                    # print(np.zeros((step, num_joints, 7)).shape)
                idx2 = pid_set.index(pid)
               # print(idx2,idx_set[idx2])
#                print(skeleton_set[idx2][idx_set[idx2]].shape)   （25，7）

                skeleton_set[idx2][idx_set[idx2]] =np.asarray([list(map(float, np.array(line_per.strip().split())[sidx]))
                                                                          for line_per in lines[start+1:start+26]])
                idx_set[idx2] = idx_set[idx2] + 1
                start = start + 26
            else:
                start = start + 1
        std_set=[] #-0.06616542 0.07068557 这串数字也不知道是什么，好像是每个文件的平均值？
        for idx2 in range(len(idx_set)):
            skeleton_set[idx2] = skeleton_set[idx2][0:idx_set[idx2]]
            if 1:
                sx = np.average(np.std(skeleton_set[idx2][:,:,0], axis=0))
                sy = np.average(np.std(skeleton_set[idx2][:,:,1], axis=0))
            else:
                xm = np.abs(skeleton_set[idx2][1:idx_set[idx2],:,0] - skeleton[0:idx_set[idx2]-1,:,0])
                sx = np.average(np.std(xm, axis=0))
                ym = np.abs(skeleton[1:idx_set[idx2],:,1] - skeleton[0:idx_set[idx2]-1,:,1])
                sy = np.average(np.std(ym, axis=0))
            std_set.append((sx, sy))

        return skeleton_set, pid_set, std_set

    def save_h5_file_seq(self, save_home, trn_list, trn_label, split='train', num_sample_save = 10000, num_seq=100):
        skeleton_list = []
        label_list = []
        seq_len_list = []
        iter_idx = 0
        save_idx = 0
        for idx, name in enumerate(trn_list):
            # load and sample skeleton
            skeleton = self.load_skeleton_file(name)
            # only use postion or angele
            skeleton = skeleton[:,:, 0:3]
            # skeleton = skeleton[:,:, 3:7]
            if skeleton.shape[0] < num_seq:
                # pad zeros in front of skeleton data
                sample = np.concatenate((np.zeros((num_seq-skeleton.shape[0], skeleton.shape[1], skeleton.shape[2])),
                         skeleton), axis=0)
            else:
                sidx = np.arange(num_seq)
                sample = skeleton[sidx]

            seq_len_list.append(skeleton.shape[0])
            skeleton_list.append(sample)
            label_list.append(trn_label[idx])

            iter_idx = iter_idx + 1
            if iter_idx== num_sample_save:
                # save skeleton
                skeleton_list = np.asarray(skeleton_list, dtype='float32')
                label_list = np.asarray(label_list, dtype='float32')
                seq_len_list = np.asarray(seq_len_list)
                save_name = os.path.join(save_home, 'seq' + str(num_seq) + '_' + split + str(save_idx) + '.h5')
                with h5py.File(save_name, 'w') as f:
                    f['data'] = skeleton_list
                    f['label'] = label_list
                    f['seq_len_list'] = seq_len_list
                save_idx = save_idx + 1
                iter_idx = 0
                skeleton_list = []
                label_list = []
                seq_len_list = []
        if iter_idx > 0:
            skeleton_list = np.asarray(skeleton_list, dtype='float32')
            label_list = np.asarray(label_list, dtype='float32')
            seq_len_list = np.asarray(seq_len_list)
            save_name = os.path.join(save_home, 'seq' + str(num_seq) + '_' + split + str(save_idx) + '.h5')
            with h5py.File(save_name, 'w') as f:
                f['data'] = skeleton_list
                f['label'] = label_list
                f['seq_len_list'] = seq_len_list
    
    def calculate_height(self, skeleton):
        center1 = (skeleton[:,2,:] + skeleton[:,8,:] + skeleton[:,4,:] + skeleton[:,20,:])/4
        w1 = skeleton[:,23,:] - center1
        w2 = skeleton[:,22,:] - center1
        center2 = (skeleton[:,1,:] + skeleton[:,0,:] + skeleton[:,16,:] + skeleton[:,12,:])/4
        h0 = skeleton[:,3,:] - center2
        h1 = skeleton[:,19,:] - center2
        h2 = skeleton[:,15,:] - center2
        width = np.max([np.max(np.abs(w1[:,0])), np.max(np.abs(w2[:,0]))])
        heigh1 = np.max(h0[:,1])
        heigh2 = np.max([np.max(np.abs(h1[:,1])), np.max(np.abs(h2[:,1]))])
        return np.asarray([width, heigh1, heigh2])
                
    def caculate_person_height(self, h5_file, list_file):
        # average value of different person: 0.36026082  0.61363413  0.76827  (mean for each person)
        # average value of different person: 1.67054954  0.87844846  1.28303429 (max for each person)
        # average value of different person: 0.0680575   0.19834167  0.21219039 (min for each person)
        name_list = [line.strip() for line in open(list_file, 'r').readlines()]
        pid_list = np.array([int(name[9:12]) for name in name_list])
        with h5py.File(h5_file,'r') as hf:
            wh_set = []
            for pid in set(pid_list):
                sidx = np.where(pid_list==pid)[0]
                wh = np.zeros((len(sidx), 3))
                for i, idx in enumerate(sidx):
                    name = name_list[idx]
                    skeleton = np.asarray(hf.get(name))
                    wh[i] = self.calculate_height(skeleton)
                wh_set.append(wh.max(axis=0)) # notice: mean or max for different position, view points
            wh_set = np.asarray(wh_set)
            print (wh_set.mean(axis=0))

if __name__ == '__main__':
    data_path = '../../nturgb+d_skeletons/'
    db = ntu_rgbd(data_path)
    subj=False
    if subj:
        trn_list, tst_list = db.cross_subject_split()
        db.save_h5_file_skeleton_list('data/subj_seq', trn_list, split='train')
        db.save_h5_file_skeleton_list('data/subj_seq', tst_list, split='test')
    else:
        trn_list, tst_list = db.cross_view_split()
        db.save_h5_file_skeleton_list('data/view_seq', trn_list, split='train')
        db.save_h5_file_skeleton_list('data/view_seq', tst_list, split='test')

    

