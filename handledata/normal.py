# -*- coding: utf-8 -*-
import numpy as np
import csv
from get_data import get_data

file1 = '/Users/gy/Documents/learn/maching_learning/mynote/mapper out/data/forpaper/winequality/winequality-white.csv'
file2 = '/Users/gy/Documents/learn/maching_learning/mynote/mapper out/data/forpaper/winequality/winequality-red.csv'
file3 = '/Users/gy/Documents/learn/maching_learning/mynote/mapper out/data/forpaper/winequality/whiter&red.csv'
new_file = '/Users/gy/Documents/learn/maching_learning/mynote/mapper out/data/forpaper/winequality/whiter&red-normal.csv'

def normal_data(temp_data):
    std = np.std(temp_data, axis=0)
    print('std:')
    print(std)
    means = np.mean(temp_data, axis=0)
    print('means:')
    print(means)
    temp_data = (temp_data-means) / std
    return temp_data


def write_data(new_file, data):
    csvfile = open(new_file, 'wb')
    writer = csv.writer(csvfile)
    writer.writerows(data)

    csvfile.close()

if __name__ == '__main__':

    data1 = get_data(file1)
    data1 = normal_data(data1)
    data2 = get_data(file2)
    data2 = normal_data(data2)
    data = get_data(file3)
    data = normal_data(data)
    write_data(new_file, data)


'''
设置单元格样式
'''

'''
def set_style(name, height, bold=False):
    style = xlwt.XFStyle()  # 初始化样式

    font = xlwt.Font()  # 为样式创建字体
    font.name = name  # 'Times New Roman'
    font.bold = bold
    # f.underline= Font.UNDERLINE_DOUBLE
    font.color_index = 4
    font.height = height

    # borders= xlwt.Borders()
    # borders.left= 6
    # borders.right= 6
    # borders.top= 6
    # borders.bottom= 6

    style.font = font
    # style.borders = borders

    return style
'''



'''
# 写excel
def write_excel():
    f = xlwt.Workbook(encoding="utf-8")  # 创建工作簿



    Data1 = f.add_sheet(u'Data1', cell_overwrite_ok=True)  # 创建sheet
    for i in range(60):
        if i<20:

            Data1.write(i,0,float(1))
            Data1.write(i,1,float(i*0.5))
        elif i<40:
            Data1.write(i,0,float(i*0.25))
            Data1.write(i,1,float(1))
        else:
            m=(i-40)*0.5
            Data1.write(i,0,float(m))
            n=11-m
            Data1.write(i,1,float(n))


    f.save('Data1.csv')  # 保存文件


if __name__ == '__main__':

    write_excel()
'''