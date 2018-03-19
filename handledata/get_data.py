# -*- coding: utf-8 -*-
import numpy as np
import csv

#从文件读入原始数据，转化为数组返回
def get_mldata(file, shulle=False):
    temp = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            temp.append(row)
    temp = np.array(temp, float)
    y = []
    for j in temp[:,-1]:
        label = [0, 0]
        label[int(j)] = 1
        y.append(label)

    if shulle == True:

        shulle_x= []
        shulle_y= []


        idx = np.random.permutation(temp.shape[0])
        for i in idx:
            shulle_x.append(temp[i, 0:-1].tolist())
            shulle_y.append(y[i])
        x = np.array(shulle_x)
        y = np.array(shulle_y)
    else:
        x = temp[:, 0:-1]
        y = np.array(y)
    return x, y, temp[:, -1]


def get_data(file, shulle=False):
    temp = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            temp.append(row)
    temp = np.array(temp, float)
    print('total number %d' % temp.shape[0])
    if shulle == True:
        idx = np.random.permutation(temp.shape[0])
        data = []
        for i in idx:
            data.append(temp[i].tolist())
        return np.array(data)
    else:

        return temp



