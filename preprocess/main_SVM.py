import SVM.data_preprocessor
import csv
from sklearn import svm
import sklearn
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.layers import LeakyReLU


def train_by_SVM():
    with open('dataset/Fall2_Cam5.avi_keys.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')  # change contents to floats
        # get header from first row
        header = next(reader)
        # get all the rows as a list
        data = list(reader)
        # transform data into numpy array
        data = np.array(data).astype(float)

    x, y = np.split(data, (37,), axis=1)
    x = x[:, :36]
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=1, train_size=0.6)
    clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())
    print(clf.score(x_train, y_train))  # 精度
    y_hat = clf.predict(x_train)
    # show_accuracy(y_hat, y_train, '训练集')
    print(clf.score(x_test, y_test))
    y_hat = clf.predict(x_test)
    # show_accuracy(y_hat, y_test, '测试集')

    print('decision_function:\n', clf.decision_function(x_train))
    print('\npredict:\n', clf.predict(x_train))


SVM.data_preprocessor.generate_dataset('../../fall_keys_coco/falls_keys/')
