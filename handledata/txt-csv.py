# -*- coding: utf-8 -*-：
import csv
txt_file = "/Users/gy/Documents/learn/maching_learning/mynote/mapper out/data/forpaper/banknote/data_banknote_authentication.txt"
csv_file = "/Users/gy/Documents/learn/maching_learning/mynote/mapper out/data/forpaper/banknote/data_banknote_authentication.csv"

in_txt = csv.reader(open(txt_file, 'r'), delimiter=',', escapechar='\n')
out_csv = csv.writer(open(csv_file, 'w'))
out_csv.writerows(in_txt)
