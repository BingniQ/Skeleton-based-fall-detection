import os
import json
import csv
import numpy as np
import pandas as pd
import datetime
import re
import time
import itertools


class preprocess(object):
    def __init__(self, data_path, pathout):
        self._path_in = data_path
        self._path_out = pathout

    def get_all_names(self):  # ../../fall_keys_coco/falls_keys/
        filenames = os.listdir(self._path_in)  # Fall1_Cam1.avi_keys
        list_all_names = []
        for filename in filenames:  # loop through all the files and folders
            file_name = filename.split(".")[0]  # Fall1_Cam1
            list_all_names.append(file_name)
        return list_all_names

    def train_test_split(self):
        list_all_names = os.listdir(self._path_in)
        train_id = [1, 4, 5]
        train_list = []
        for file in list_all_names:
            cam = re.match(r"([a-z]+)([0-9]+)", file.split('_')[1], re.I)
            cam_id = int(cam[2])
            if cam_id in train_id:
                train_list.append(file)
        test_list = list(set(list_all_names) - set(train_list))
        print(test_list)
        return train_list, test_list  # Fall1_Cam1.avi_keys

    def calculate_time(self, x):
        seconds = datetime.timedelta(hours=x.hour, minutes=x.minute, seconds=x.second).total_seconds()
        return seconds

    def get_fall_time(self, video_name):
        action = re.match(r"([a-z]+)([0-9]+)", video_name.split('_')[0], re.I)
        action_name = action[1]
        action_id = int(action[2])
        time_result = []
        if action_name == 'Fall':
            fall_data = pd.read_excel('../../Data_Description.xlsx', sheet_name='Fall')
            # check fall time period for each file
            for i, row in fall_data.iterrows():
                if row['Unnamed: 0'] == action_id:
                    time_start = self.calculate_time(row['Annotate'])
                    time_fall = self.calculate_time(row['Unnamed: 9'])
                    time_end = self.calculate_time(row['Unnamed: 10'])
                    time_result.append(time_start)
                    time_result.append(time_fall)
                    time_result.append(time_end)
        else:
            time_result.append([0, 0, 0])
        return time_result

    def get_JSON_file_list(self, video_name):  # Fall1_Cam1.avi_keys
        json_files_list = [pos_json for pos_json in os.listdir(self._path_in + video_name + '/') if
                           pos_json.endswith('.json')]
        return json_files_list  # Fall1_Cam1_000000000934(12位)_keypoints.json

    def get_frame_list(self, video_name):
        json_files = self.get_JSON_file_list(video_name)
        frame_list = []
        for file in json_files:
            frame = int(file.split('_')[2])
            frame_list.append(frame)
        return frame_list

    def get_nrOfPerson_list(self, video_name):  # Fall1_Cam1.avi_keys
        json_files = self.get_JSON_file_list(video_name)
        list_nrOfPerson = []
        for index, js in enumerate(json_files):
            with open(os.path.join(self._path_in + video_name + '/', js)) as json_file:
                json_text = json.load(json_file)
                people = json_text['people']
                list_nrOfPerson.append(len(people))
        return list_nrOfPerson  # number of people in Fall1_Cam1_000000000934(12位)_keypoints.json

    def get_skeleton_list(self, video_name):  # skeleton list for Fall1_Cam1.avi_keys
        json_files = self.get_JSON_file_list(video_name)
        skeleton_per_video = []
        for index, js in enumerate(json_files):
            with open(os.path.join(self._path_in + video_name + '/', js)) as json_file:
                json_text = json.load(json_file)
                people = json_text['people']
                nrOfPeople = len(people)
                if nrOfPeople != 0:
                    skeleton_per_frame = []
                    i = 1
                    while i <= nrOfPeople:
                        pose_keypoints_2d = people[i - 1]['pose_keypoints_2d']
                        a = np.array(pose_keypoints_2d)
                        b = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13,
                             15, 16, 18, 19, 21, 22, 24, 25, 27, 28,
                             30, 31, 33, 34, 36, 37, 39, 40, 42, 43,
                             45, 46, 48, 49, 51, 52]
                        inputData = list(a[b])
                        skeleton_per_frame.append(inputData)
                        i = i + 1
                    skeleton_per_video.append(skeleton_per_frame)
                else:
                    pose_keypoints_2d = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
                    # pose_keypoints_2d=[]
                    skeleton_per_video.append(pose_keypoints_2d)
        # print(skeleton_per_video[0])
        # print(skeleton_per_video[890])  # should be two people
        return skeleton_per_video  # list

    def get_continues_frame_postion(self, person_list, frame_list):
        merge_list = list(zip(person_list, frame_list))
        group_list = []
        for key, group in itertools.groupby(merge_list, lambda x: x[0]):
            temp_list = list(group)
            length = len(temp_list)
            start_frame = (temp_list[0])[1]
            end_frame = (temp_list[length - 1])[1]
            temp = [length, key, start_frame, end_frame]
            group_list.append(temp)
        # print(group_list)
        return group_list

    def filter_nrOfPerson(self, person_list, frame_list):
        group_list = self.get_continues_frame_postion(person_list, frame_list)
        print("origin:", group_list)

        key_list = []
        for key, group in itertools.groupby(person_list):
            if len(list(group)) >= 60 and key != 0:
                key_list.append(key)

        if key_list != []:
            nrOfPerson = max(key_list)
            print("max number of people in this video is: ", nrOfPerson)

            max_nrOfPerson = max(person_list)
            length = max([len(list(v)) for k, v in itertools.groupby(person_list) if k == max_nrOfPerson])
            # print(length)
            while length < 60:
                for item in group_list:
                    if item[1] > 1 and item[0] < 60:
                        frame_start_temp = item[2]
                        frame_end_temp = item[3]
                        while frame_start_temp <= frame_end_temp:
                            person_list[frame_start_temp] = person_list[frame_start_temp] - 1
                            frame_start_temp = frame_start_temp + 1
                max_nrOfPerson = max(person_list)
                length = max([len(list(v)) for k, v in itertools.groupby(person_list) if k == max_nrOfPerson])
                group_list = self.get_continues_frame_postion(person_list, frame_list)

            i = 1
            while i <= nrOfPerson:
                a = [len(list(v)) for k, v in itertools.groupby(person_list) if k == i]
                print(i, '连续出现的最大次数为：%d' % max(a))
                i = i + 1

        return group_list, person_list

    def get_sliced_sample(self, group_list, skeleton_list, person_list, frame_list, start_frame, end_frame):
        people = 0
        for group in group_list:
            if group[3] >= start_frame and group[3] < end_frame:
                people = group[1]
            if group[2] < end_frame and group[2] >= start_frame:
                if group[1] > people:
                    people = group[1]

        frame_temp = start_frame
        while frame_temp <= end_frame:
            while len(skeleton_list[frame_temp]) < people:
                skeleton_list[frame_temp].append(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            person_list[frame_temp] = people
            frame_temp += 1

        # slice fall samples with the center of fall
        # group_list = self.get_continues_frame_postion(person_list, frame_list)
        sliced_skeleton_set = skeleton_list[start_frame:end_frame + 1]
        return people, sliced_skeleton_set

    def generate_samples(self, video_name):
        fall_time = self.get_fall_time(video_name)
        time_start = fall_time[0]
        time_end = fall_time[2]
        frame_list = self.get_frame_list(video_name)
        person_list = self.get_nrOfPerson_list(video_name)
        skeleton_list = self.get_skeleton_list(video_name)

        if time_start != time_end:
            fall_frame_length = (time_end - time_start) * 30
            # start_fall_frame = int(time_start*30)
            # end_fall_frame = int(time_start*30+99)
            start_fall_frame = int(time_start * 30 + 1 - (100 - fall_frame_length) / 2)
            end_fall_frame = int(time_end * 30 + (100 - fall_frame_length) / 2)
            print("start fall frame:", start_fall_frame)
            print('end fall frame:', end_fall_frame)

            group_list, person_list = self.filter_nrOfPerson(person_list, frame_list)

            # fall sample
            people, skeleton_set_fall = self.get_sliced_sample(group_list, skeleton_list, person_list, frame_list,
                                                               start_fall_frame, end_fall_frame)
            print('跌落人数', people)
            if people > 1:
                while people >= 1:
                    filepath = path_out + video_name.split('.')[0] + '_' + str(start_fall_frame) + '_' + str(
                        end_fall_frame) + '_A0_P' + str(people) + '.txt'
                    with open(filepath, 'w') as file:
                        file.writelines(
                            '\t'.join(str(j) for j in i) + '\n' for i in self.column(skeleton_set_fall, people - 1))
                    print(filepath)
                    people -= 1
            elif people == 1:
                filepath = path_out + video_name.split('.')[0] + '_' + str(start_fall_frame) + '_' + str(
                    end_fall_frame) + '_A0_P' + str(people) + '.txt'
                with open(filepath, 'w') as file:
                    file.writelines(
                        '\t'.join(str(j) for j in i) + '\n' for i in self.column(skeleton_set_fall, people - 1))
                print(filepath)

            # non-fall sample
            start_frame = start_fall_frame - 100
            end_frame = end_fall_frame - 100
            while start_frame >= frame_list[0]:
                people, skeleton_set_temp = self.get_sliced_sample(group_list, skeleton_list, person_list, frame_list,
                                                                   start_frame, end_frame)
                if people > 1:
                    while people >= 1:
                        filepath = path_out + video_name.split('.')[0] + '_' + str(start_frame) + '_' + str(
                            end_frame) + '_A1_P' + str(people) + '.txt'
                        with open(filepath, 'w') as file:
                            file.writelines(
                                '\t'.join(str(j) for j in i) + '\n' for i in self.column(skeleton_set_temp, people - 1))
                        print(filepath)
                        people -= 1
                elif people == 1:
                    filepath = path_out + video_name.split('.')[0] + '_' + str(start_frame) + '_' + str(
                        end_frame) + '_A1_P' + str(people) + '.txt'
                    with open(filepath, 'w') as file:
                        file.writelines(
                            '\t'.join(str(j) for j in i) + '\n' for i in self.column(skeleton_set_temp, people - 1))
                    print(filepath)
                start_frame -= 100
                end_frame -= 100

            start_frame_after = start_fall_frame + 100
            end_frame_after = end_fall_frame + 100
            while end_frame_after <= frame_list[len(frame_list) - 1]:
                people, skeleton_set_temp = self.get_sliced_sample(group_list, skeleton_list, person_list, frame_list,
                                                                   start_frame_after, end_frame_after)
                if people > 1:
                    while people >= 1:
                        filepath = path_out + video_name.split('.')[0] + '_' + str(start_frame_after) + '_' + str(
                            end_frame_after) + '_A1_P' + str(people) + '.txt'
                        with open(filepath, 'w') as file:
                            file.writelines(
                                '\t'.join(str(j) for j in i) + '\n' for i in self.column(skeleton_set_temp, people - 1))
                        print(filepath)
                        people -= 1
                elif people == 1:
                    filepath = path_out + video_name.split('.')[0] + '_' + str(start_frame_after) + '_' + str(
                        end_frame_after) + '_A1_P' + str(people) + '.txt'
                    with open(filepath, 'w') as file:
                        file.writelines(
                            '\t'.join(str(j) for j in i) + '\n' for i in self.column(skeleton_set_temp, people - 1))
                    print(filepath)
                start_frame_after += 100
                end_frame_after += 100

    def generate_all(self):
        filenames = self.get_all_names()
        for name in filenames:
            print(name)
            db.generate_samples(str(name) + '.avi_keys')

    def column(self, matrix, i):
        return [row[i] for row in matrix]

    def read_file(self, filepath):
        f = open(filepath, "r")
        contents = f.readlines()
        print(contents[0].strip().split())


if __name__ == '__main__':
    data_path = '../../fall_keys_coco/falls_keys/'
    path_out = '../../samples/'
    db = preprocess(data_path, path_out)
    # db.train_test_split() # ['Fall48_Cam2.avi_keys', 'Fall13_Cam3.avi_keys', 'Fall19_Cam3.avi_keys', 'Fall42_Cam3.avi_keys', 'Fall49_Cam2.avi_keys', 'Fall20_Cam2.avi_keys', 'Fall49_Cam3.avi_keys', 'Fall22_Cam3.avi_keys', 'Fall28_Cam2.avi_keys', 'Fall33_Cam3.avi_keys', 'Fall36_Cam2.avi_keys', 'Fall55_Cam2.avi_keys', 'Fall44_Cam2.avi_keys', 'Fall39_Cam3.avi_keys', 'Fall54_Cam3.avi_keys', 'Fall35_Cam2.avi_keys', 'Fall52_Cam3.avi_keys', 'Fall12_Cam3.avi_keys', 'Fall55_Cam3.avi_keys', 'Fall26_Cam3.avi_keys', 'Fall16_Cam3.avi_keys', 'Fall14_Cam2.avi_keys', 'Fall50_Cam2.avi_keys', 'Fall43_Cam2.avi_keys', 'Fall5_Cam2.avi_keys', 'Fall33_Cam2.avi_keys', 'Fall48_Cam3.avi_keys', 'Fall16_Cam2.avi_keys', 'Fall11_Cam3.avi_keys', 'Fall30_Cam3.avi_keys', 'Fall37_Cam3.avi_keys', 'Fall31_Cam2.avi_keys', 'Fall26_Cam2.avi_keys', 'Fall12_Cam2.avi_keys', 'Fall27_Cam3.avi_keys', 'Fall11_Cam2.avi_keys', 'Fall31_Cam3.avi_keys', 'Fall24_Cam2.avi_keys', 'Fall50_Cam3.avi_keys', 'Fall20_Cam3.avi_keys', 'Fall37_Cam2.avi_keys', 'Fall21_Cam2.avi_keys', 'Fall35_Cam3.avi_keys', 'Fall41_Cam3.avi_keys', 'Fall2_Cam3.avi_keys', 'Fall52_Cam2.avi_keys', 'Fall15_Cam3.avi_keys', 'Fall34_Cam2.avi_keys', 'Fall54_Cam2.avi_keys', 'Fall23_Cam3.avi_keys', 'Fall40_Cam3.avi_keys', 'Fall17_Cam2.avi_keys', 'Fall1_Cam2.avi_keys', 'Fall45_Cam2.avi_keys', 'Fall4_Cam2.avi_keys', 'Fall2_Cam2.avi_keys', 'Fall21_Cam3.avi_keys', 'Fall39_Cam2.avi_keys', 'Fall29_Cam2.avi_keys', 'Fall5_Cam3.avi_keys', 'Fall32_Cam2.avi_keys', 'Fall15_Cam2.avi_keys', 'Fall18_Cam3.avi_keys', 'Fall30_Cam2.avi_keys', 'Fall3_Cam2.avi_keys', 'Fall22_Cam2.avi_keys', 'Fall46_Cam3.avi_keys', 'Fall25_Cam3.avi_keys', 'Fall24_Cam3.avi_keys', 'Fall17_Cam3.avi_keys', 'Fall3_Cam3.avi_keys', 'Fall27_Cam2.avi_keys', 'Fall41_Cam2.avi_keys', 'Fall44_Cam3.avi_keys', 'Fall10_Cam2.avi_keys', 'Fall51_Cam2.avi_keys', 'Fall51_Cam3.avi_keys', 'Fall46_Cam2.avi_keys', 'Fall25_Cam2.avi_keys', 'Fall38_Cam3.avi_keys', 'Fall36_Cam3.avi_keys', 'Fall38_Cam2.avi_keys', 'Fall43_Cam3.avi_keys', 'Fall53_Cam3.avi_keys', 'Fall10_Cam3.avi_keys', 'Fall28_Cam3.avi_keys', 'Fall1_Cam3.avi_keys', 'Fall32_Cam3.avi_keys', 'Fall42_Cam2.avi_keys', 'Fall29_Cam3.avi_keys', 'Fall53_Cam2.avi_keys', 'Fall19_Cam2.avi_keys', 'Fall47_Cam3.avi_keys', 'Fall4_Cam3.avi_keys', 'Fall23_Cam2.avi_keys', 'Fall13_Cam2.avi_keys', 'Fall45_Cam3.avi_keys', 'Fall47_Cam2.avi_keys', 'Fall18_Cam2.avi_keys', 'Fall14_Cam3.avi_keys']
    # db.get_skeleton_list('Fall1_Cam1.avi_keys') # [[0.0, 0.0, 607.147, 466.326, 646.631, 438.726, 0.0, 0.0, 0.0, 0.0, 583.476, 480.842, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 593.979, 405.854, 540.033, 446.608], [697.944, 426.883, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 700.583, 415.013, 703.204, 417.693, 0.0, 0.0, 736.118, 417.672]]
    # db.generate_samples('Fall1_Cam1.avi_keys') # 1406 1505
    # db.get_frame_list('Fall1_Cam1.avi_keys') # 0~4141
    # db.generate_samples('Fall1_Cam1.avi_keys')
    # db.generate_samples('Fall30_Cam5.avi_keys')
    db.generate_all()




def store_json_to_dataset(filename, dataset_name, time_start, time_end):
    path_to_json = filename
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]


    output_file_name=dataset_name.split(".")[0]+'_'+str(time_start)+'_'+str(time_end) + '.csv'

    print(output_file_name)

    with open('../../dataset/' + output_file_name, 'w', newline='') as outf:
        writer = csv.writer(outf, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        header = ['Nose_x', 'Nose_y', 'Neck_x', 'Neck_y', 'RShoulder_x', 'RShoulder_y', 'RElbow_x', 'RElbow_y',
                  'RWrist_x',
                  'RWrist_y', 'LShoulder_x', 'LShoulder_y', 'LElbow_x', 'LElbow_y', 'LWrist_x', 'LWrist_y', 'RHip_x',
                  'RHip_y', 'RKnee_x', 'RKnee_y', 'RAnkle_x', 'RAnkle_y', 'LHip_x', 'LHip_y', 'LKnee_x', 'LKnee_y',
                  'LAnkle_x', 'LAnkle_y', 'REye_x', 'REye_y', 'LEye_x', 'LEye_y', 'REar_x', 'REar_y', 'LEar_x',
                  'LEar_y',
                  'frame', 'class']
        writer.writerow(header)

        for index, js in enumerate(json_files):

            with open(os.path.join(path_to_json, js)) as json_file:
                json_text = json.load(json_file)
                people = json_text['people']

                if len(people) != 0:
                    if(len(people)>1):
                        print(len(people),js)
                    pose_keypoints_2d = people[0]['pose_keypoints_2d']
                    a = np.array(pose_keypoints_2d)
                    b = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34, 36,
                         37,
                         39, 40, 42, 43, 45, 46, 48, 49, 51, 52]
                    inputData = list(a[b])
                    inputData.append(index)
                    if index >= 30 * time_start and index <= 30 * time_end:
                        inputData.append(1)
                    else:
                        inputData.append(0)
                    writer.writerow(inputData)
                else:
                    pose_keypoints_2d = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print("Success generate dataset: " + filename)
    return 0


def calculate_time(x):
    seconds = datetime.timedelta(hours=x.hour, minutes=x.minute, seconds=x.second).total_seconds()
    return seconds


def get_fall_time(scenario_number):
    # Load a sheet into a DataFrame by name: fall_data
    fall_data = pd.read_excel('../../Data_Description.xlsx', sheet_name='Fall')
    time_result = []
    # check fall time period for each file
    for i, row in fall_data.iterrows():
        if row['Unnamed: 0'] == scenario_number:
            time_start = calculate_time(row['Annotate'])
            time_fall = calculate_time(row['Unnamed: 9'])
            time_end = calculate_time(row['Unnamed: 10'])
            time_result.append(time_start)
            time_result.append(time_fall)
            time_result.append(time_end)
            return time_result


# path can be '../coco_keys/'
def generate_dataset(path):
    # get all files' and folders' names in the current directory
    filenames = os.listdir(path)
    for filename in filenames:  # loop through all the files and folders
        result = re.match(r"([a-z]+)([0-9]+)", filename, re.I)
        action = result[1]
        scenario = int(result[2])
        if action == 'Fall':
            fall_time = get_fall_time(scenario)
            print(action, scenario, fall_time)
            store_json_to_dataset(path + filename + '/', filename, fall_time[0], fall_time[2])
        if action!='Fall':
            store_json_to_dataset(path+filename + '/', filename, 0, 0)
    return 0


# path_in should be '../dataset/xxx' path_out should be '../edit_dataset/xxxA1'
def generate_samples(path_in,path_out):
    filenames = os.listdir(path_in)
    print(filenames)
    for filename in filenames:
        filename = os.path.splitext(filename)[0]
        newstr = ''.join((ch if ch in '0123456789.-e' else ' ') for ch in filename)
        listOfNumbers = [float(i) for i in newstr.split()]
        time_start = listOfNumbers[2]
        time_end = listOfNumbers[3]
        fall_frame_length = (time_end-time_start)*30
        start_frame = int(time_start*30-(100 - fall_frame_length)/2)
        end_frame = int(time_end*30+(100 - fall_frame_length)/2)
        print(start_frame, end_frame)

        # with open(os.path.join(path_in+filename)) as infile:
        with open("../../dataset/Fall1_Cam1_47.0_50.0.csv") as infile:
            reader = csv.DictReader(infile)
            header = reader.fieldnames
            rows = [row for row in reader]
            row_count = len(rows)
            for index, row in rows:
                result = row['class'] # 0 for non-fall, 1 for fall
                frame = row['frame']
                if frame >= start_frame and frame <= end_frame:
                    if rows[index]['frame']+1 == row[index+1]['frame']:
                        print("continues")



        #     start_index = 0
        #     # here, we slice the total rows into pages, each page having [row_per_csv] rows
        #     while start_index < row_count:
        #         pages.append(rows[start_index: start_index+rows_per_csv])
        #         start_index += rows_per_csv
        #
        #     for i, page in enumerate(pages):
        #         with open('{}_{}.csv'.format(file_name, i+1), 'w+') as outfile:
        #             writer = csv.DictWriter(outfile, fieldnames=header)
        #             writer.writeheader()
        #             for row in page:
        #                 writer.writerow(row)
        #
        #         print('DONE splitting {} into {} files'.format(filename, len(pages)))
        #
        #
        #
        # with open(os.path.join(path_in+filename)) as csv_in:
        #     with open(path_out + filename, 'w', newline='') as csv_out:
        #         writer = csv.writer(csv_out)
        #         for row in csv.reader(csv_in):
        #             if row[37] == 1 and row[36]==time_start*30:
        #                 writer.writerow(row)


def check_continuity(path_in):
    filenames = os.listdir(path_in)






    # store_json_to_dataset("../../fall_keys_coco/falls_keys/Fall1_Cam1.avi_keys/",'Fall1_Cam1.avi_keys',47.0,50.0)
    # # generate_dataset('../../fall_keys_coco/falls_keys')
    # # generate_samples('../../dataset/','../../samples/')
    # filename = "Fall1_Cam1_47.0_50.0.csv"
    # filename = os.path.splitext(filename)[0]
    # newstr = ''.join((ch if ch in '0123456789.-e' else ' ') for ch in filename)
    # listOfNumbers = [float(i) for i in newstr.split()]
    # time_start = listOfNumbers[2]
    # time_end = listOfNumbers[3]
    # fall_frame_length = (time_end-time_start)*30
    # start_fall_frame = int(time_start*30+1-(100 - fall_frame_length)/2)
    # end_fall_frame = int(time_end*30+(100 - fall_frame_length)/2)
    # print(start_fall_frame, end_fall_frame)
    #
    #
    # # with open(os.path.join(path_in+filename)) as infile:
    # with open("../../dataset/Fall1_Cam1_47.0_50.0.csv") as infile:
    #     reader = csv.DictReader(infile)
    #     header = reader.fieldnames
    #     rows = [row for row in reader]
    #     row_count = len(rows)
    #     start_frame = int(rows[0]['frame'])
    #     end_frame = int(rows[row_count-1]['frame'])
    #     print(start_frame,end_frame)
    #     before_nrOfSamples=int((start_fall_frame-start_frame)/100)
    #     after_nrOfSamples= int((end_frame-end_fall_frame)/100)
    #     print(before_nrOfSamples,after_nrOfSamples)
    #
    #     samples_non_fall=np.zeros(before_nrOfSamples+after_nrOfSamples)
    #     sample_fall=[]
    #
    #     for index, row in enumerate(rows):
    #         result = int(row['class']) # 0 for non-fall, 1 for fall
    #         frame = int(row['frame'])
    #         sample_temp=[]
    #         before_count=1
    #
    #         # check if frame is continues
    #         if int(rows[index]['frame'])+1 == int(rows[index+1]['frame']):
    #             sample_temp.append(row)
    #         else:
    #             print("next frame is not continues")
    #             print(filename, index, frame)
    #             exit()


            # while before_count<= before_nrOfSamples:
            # if frame >= start_fall_frame-100*before_count and frame <= end_fall_frame-100*before_count:
            #     if int(rows[index]['frame'])+1 == int(rows[index+1]['frame']):
            #         sample_temp.append(row)
            #     else:
            #         print("next frame is not continues")
            #         print(filename, index, frame)
            #         exit()
            #
            # if frame >= start_fall_frame and frame <= end_fall_frame:
            #     if int(rows[index]['frame'])+1 == int(rows[index+1]['frame']):
            #         sample_fall.append(row)
            #     else:
            #         print("next frame is not continues")
            #         print(filename, index, frame)
            #         exit()

        # array_temp=np.array(sample_temp)
        # print(sample_temp)
        # print(array_temp)
        # samples_non_fall[before_count-1]=array_temp
        # before_count=before_count+1

    # print(samples_non_fall)





# def generate_svm_dataset(path):
#     global index
#     global video_canvas_img, img
#     global x_old, y_old, x_old_speed, y_old_speed, x_speed, y_speed
#     global head_x_old, head_y_old, head_x_old_speed, head_x_speed, head_y_old_speed, head_y_speed
#     global enable
#     global old_frame
#
#     path_to_json = path
#     json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
#     for index, js in enumerate(json_files):
#         with open(os.path.join(path_to_json, js)) as f:
#             if f == None:
#                 return
#             data = json.load(f)
#             x = [0] * 18
#             y = [0] * 18
#             head_count = 0
#             if data['people']:
#                 for i in range(0, 18):
#                     x[i] = data['people'][0]['pose_keypoints_2d'][3 * i]
#                     y[i] = data['people'][0]['pose_keypoints_2d'][3 * i + 1]
#
#             # Take neck to represent the body
#             x_now = x[1]
#             y_now = y[1]
#             # Also take head into consideration, first count valid key pints of head, then compute their mean position
#             if x[0]:
#                 head_count = head_count + 1
#             for i in range(0, 4):
#                 if x[i + 14]:
#                     head_count = head_count + 1
#             # Calculate head position
#             head_x_now = 0
#             head_y_now = 0
#             if head_count:
#                 head_x_now = (x[0] + x[14] + x[15] + x[16] + x[17]) / head_count
#                 head_y_now = (y[0] + y[14] + y[15] + y[16] + y[17]) / head_count
#
#             # Calculate speed
#             x_old_speed = x_speed
#             y_old_speed = y_speed
#             head_x_old_speed = head_x_speed
#             head_y_old_speed = head_y_speed
#             x_speed = x_now - x_old
#             y_speed = y_now - y_old
#             head_x_speed = head_x_now - head_x_old
#             head_y_speed = head_y_now - head_y_old
#             # old_speed = speed
#             # speed = math.sqrt((x_now - x_old)*(x_now - x_old) + (y_now - y_old)*(y_now - y_old))
#             x_average = (x_speed + x_old_speed) / 2
#             y_average = (y_speed + y_old_speed) / 2
#             head_x_average = (head_x_speed + head_x_old_speed) / 2
#             head_y_average = (head_y_speed + head_y_old_speed) / 2
#
#             print("Frame:" + str(index) + " x_speed: " + str(x_average))
#             print("Frame:" + str(index) + " y_speed: " + str(y_average))
#             print("Frame:" + str(index) + " headx_speed: " + str(head_x_average))
#             print("Frame:" + str(index) + " heady_speed: " + str(head_y_average))
#             print(" ")
#
#             # Update the position
#             x_old = x_now
#             y_old = y_now
#             head_x_old = head_x_now
#             head_y_old = head_y_now

            # # Update distance
            # distance = Distance(x, y)
            # Fall judge
            # fall(x_average, y_average, head_x_average, head_y_average, index)




# def pose_normalization(x):
#     def retrain_only_body_joints(x_input):
#         x0 = x_input.copy()
#         x0 = x0[2:2+13*2]
#         return x0
#
#     def normalize(x_input):
#         # Separate original data into x_list and y_list
#         lx = []
#         ly = []
#         N = len(x_input)
#         i = 0
#         while i<N:
#             lx.append(x_input[i])
#             ly.append(x_input[i+1])
#             i+=2
#         lx = np.array(lx)
#         ly = np.array(ly)
#
#         # Get rid of undetected data (=0)
#         non_zero_x = []
#         non_zero_y = []
#         for i in range(int(N/2)):
#             if lx[i] != 0:
#                 non_zero_x.append(lx[i])
#             if ly[i] != 0:
#                 non_zero_y.append(ly[i])
#         if len(non_zero_x) == 0 or len(non_zero_y) == 0:
#             return np.array([0] * N)
#
#         # Normalization x/y data according to the bounding box
#         origin_x = np.min(non_zero_x)
#         origin_y = np.min(non_zero_y)
#         len_x = np.max(non_zero_x) - np.min(non_zero_x)
#         len_y = np.max(non_zero_y) - np.min(non_zero_y)
#         x_new = []
#         for i in range(int(N/2)):
#             if (lx[i] + ly[i]) == 0:
#                 x_new.append(-1)
#                 x_new.append(-1)
#             else:
#                 x_new.append((lx[i] - origin_x) / len_x)
#                 x_new.append((ly[i] - origin_y) / len_y)
#         return x_new
#
#     x_body_joints_xy = retrain_only_body_joints(x)
#     x_body_joints_xy = normalize(x_body_joints_xy)
#     return x_body_joints_xy




