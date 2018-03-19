# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import csv
from get_data import get_data
import random
import math
import os
import matplotlib.pyplot as plt


filePath0 = '/Users/gy/Desktop/paperPicture/testor/temp/banknote0test.csv'
filePath1 = '/Users/gy/Desktop/paperPicture/testor/temp/banknote1test.csv'
newfile0 = '/Users/gy/Desktop/paperPicture/testor/temp/banknote0testmul0.8.csv'
newfile1 = '/Users/gy/Desktop/paperPicture/testor/temp/banknote1testmul0.8.csv'
combine_dir = '/Users/gy/Desktop/paperPicture/combine/combine5mul0.8/modify0'


#改变数据集第k列的符号
def change_syboml(data, k):
    n = data.shape[0]
    print(data.shape)
    for i in range(n):
        data[i][k] = np.fabs(data[i][k])
    return data

#将数据集整个乘以k
def mult(data, k):
    data = k * data
    #data = data + 0.3
    return data

#将两个数据集结合，k决定结合的比例
def combine(data0, data1, k):
    if k<1:
        n = data0.shape[0]
        n = int(n * k)
        temp = np.zeros(data0.shape)
        temp[0:n, :] = data0[0:n, :]
        temp[n:, :] = data1[n:, :]
        return temp
    else:
        n, d = data0.shape
        temp1 = np.zeros((2*n, d))
        temp1[0:n, :] = data0
        temp1[n:, :] = data1
        return temp1

#从原来的数据中抽取一部分, 取p
def sample(data, p):
    n, d = data.shape
    n_sample = int(n * p)
    index = random.sample(range(n), n_sample)
    temp = []
    for i in index:
        temp.append(data[i])
    temp = np.array(temp)
    return temp

def change0(data):
    new_data = data
    new_data[:, 0:-1] = -0.8 * data[:, 0:-1]
    t1 = new_data[:, 1].tolist()
    t2 = new_data[:, 2].tolist()
    new_data[:, 1], new_data[:, 2] = t2, -np.array(t1)
    #new_data[:, 3] = -new_data[:, 3]
    new_data[:, -1] = data[:, -1] + 1
    return new_data

def change1(data):
    new_data = data
    new_data[:, 0:-1] = -0.8 * data[:, 0:-1]
    new_data[:, 2] = - new_data[:, 2]
    new_data[:, 3] = - new_data[:, 3]

    t1 = new_data[:, 1].tolist()
    t2 = (new_data[:, 2]).tolist()
    new_data[:, 1] = t2
    new_data[:, 2] = t1
    new_data[:, -1] = data[:, -1] - 1
    print np.mean(new_data)
    return new_data

#写修改后的数据到新文件
def write_data(new_file, data):
    csvfile = open(new_file, 'wb')
    writer = csv.writer(csvfile)
    writer.writerows(data)

def runcombine(data0, data1):
    for i in range(9):
        k = (i + 1.) / 10
        print k
        new_data = combine(data0, data1, k)
        out_path = os.path.join('%s/1modify%.1f.csv' % (combine_dir, k))
        write_data(out_path, new_data)


def randomchange(data0, data1):
    mean0 = np.mean(data0[:, 0:-1], axis=0)
    mean1 = np.mean(data1[:, 0:-1], axis=0)
    dis_mean = mean0 - mean1
    data1[:, 0:-1] += dis_mean
    newmean1 = np.mean(data1[:, 0:-1], axis=0)
    return data1

def histogram(file):
    data = get_data(file, False)
    label = data[:, -1]
    for i in range(10):
        index = i+3000
        plt.bar(range(3000), height=label[i:index])
        print np.sum(label[i:index])
        plt.show()


def transform(file):
    data = get_data(file,False)
    data = data/(data.max()-data.min())
    data = data * 8
    return data


if __name__ == '__main__':
    #dataset0 = get_data(newfile0)
    #dataset1 = get_data(filePath1)
    #write_data(newfile0, change0(dataset0))
    #write_data(newfile1, change1(dataset1))
    #newdata = mult(dataset0, 2)
    #runcombine(dataset0, dataset1)
    #newdata = change0(dataset0)
    #write_data(newfile1, newdata)
    #new_data1 = randomchange(dataset0, dataset1)
    #write_data(newfile0, newdata)\

    #######draw histogram##########
    #histogram('/Users/gy/Documents/learn/maching_learning/mynote/mapper out/data/forpaper/credit card/default.csv')
    newdata = transform('/Users/gy/Documents/learn/maching_learning/mynote/mapper out/data/forpaper/credit card/creditcard6c3.csv')
    write_data('/Users/gy/Documents/learn/maching_learning/mynote/mapper out/data/forpaper/credit card/creditcard6c3normal.csv', newdata)