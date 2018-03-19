# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression
from handledata.get_data import get_data
import numpy as np
import os

train_file = '/Users/gy/Documents/learn/maching_learning/mynote/mapper out/data/forpaper/banknote/train/banknotetrain.csv'
test_file = '/Users/gy/Documents/learn/maching_learning/mynote/mapper out/data/forpaper/banknote/banknote1/banknote1testmodify.csv'

test_dir = '/Users/gy/Desktop/paperPicture/combine/combine3/momean'

train_data = get_data(train_file, True)
#test_data = get_data(test_file)

train_x , train_y = train_data[:, 0:-1], train_data[:, -1]
#test_x, test_y = test_data[:, 0:-1], test_data[:, -1]

lclf = LogisticRegression(penalty='l2', solver='newton-cg', C=100, max_iter=1000, tol=1e-8)
lclf.fit(train_x, train_y)


def test(x, y):

    pred_y = lclf.predict(x)

    acc = np.mean(np.equal(pred_y, y))

    return acc


#orig_acc = test(test_x, test_y)
#modi_acc = test(-0.75*test_x+2, test_y)
#modi_all = test(modify_x2, test_y)

for path in os.listdir(test_dir):
    if path !='.DS_Store':

        test_path = os.path.join('%s/%s' % (test_dir, path))
        test_data = get_data(test_path)
        test_x, test_y = test_data[:, 0:-1], test_data[:, -1]
        orig_acc0 = test(test_x, test_y)

        print('the path is %s' % path)
        print('the orignal acc is %.4f' % orig_acc0)



#print('the orignal test acc is %.4f' % orig_acc)



#print('the modify test acc is %.4f' % modi_acc)


#print('the total modify acc is %.4f' % modi_all)
