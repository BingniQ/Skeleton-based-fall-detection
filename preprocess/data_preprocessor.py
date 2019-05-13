import os
import json
import csv
import numpy as np
import pandas as pd
import datetime
import re
import time


def store_json_to_dataset(filename, dataset_name, time_start, time_end):
    path_to_json = filename
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

    with open('../../dataset/' + dataset_name + '.csv', 'w', newline='') as outf:
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
def filter_frames(path_in,path_out,time_start, time_end):
    with open(os.path.join(path_in)) as csv_in, open(os.path.join(path_out), 'wb') as csv_out:
        writer = csv.writer(csv_out)
        for row in csv.reader(csv_in):
            if row[37] == 1 and row[36]==time_start*30:
                writer.writerow(row)



if __name__ == '__main__':
    generate_dataset('../../fall_keys_coco/falls_keys/')



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




