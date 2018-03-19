# -*- coding: utf-8 -*-
from sklearn import svm
from handledata.get_data import get_data
import numpy as np
import os
from ROC_SVM import Roc
import matplotlib.pyplot as plt

train_file = '/Users/gy/Documents/learn/maching_learning/mynote/mapper out/data/forpaper/banknote/train/banknotetrain.csv'
test_file11 = '/Users/gy/Desktop/paperPicture/testmo/temp/banknote1testmul0.96.csv'
test_file1 = '/Users/gy/Desktop/paperPicture/testor/temp/aug11/banknote1test.csv'

test_dir = '/Users/gy/Desktop/paperPicture/testmo/temp'

train_data = get_data(train_file, True)


train_x, train_y = train_data[:, 0:-1], train_data[:, -1]

#modify_x0 = -0.65 * test_x0 + 0.5
#modify_x1 = -0.35 * test_x1 +0.9


clf = svm.SVC(C=1, probability=True, gamma=0.0001)
clf.fit(train_x, train_y)

#w = clf.support_vectors_
#indx = clf.support_

def test(x, y):

    pred_y = clf.predict(x)
    pred_pro = clf.predict_proba(x)

    acc = np.mean(np.equal(pred_y, y))
    #fpr, tpr, roc_auc, precision, recall = Roc(y, pred_pro[:, 1])

    #找到predy和y为0, 1 的位置，
    pre_id0 = np.array(np.where(pred_y==0)[0])

    or_id1 = np.array(np.where(y==1)[0])
    #print pre_id0
    #print or_id1
    falseNb = 0
    for l in pre_id0:
        if l in or_id1:
            falseNb += 1

    print falseNb
    if or_id1.shape[0] !=0:
        F_G = np.float32(falseNb) / or_id1.shape[0]
    else:
        F_G = 0.0


    return acc, F_G

#ds = clf.decision_function(w)

def finaltest(test_dir):
    i = 0
    name_lis = []
    fpr_lis = []
    tpr_lis = []
    rocauc_lis = []
    for path in os.listdir(test_dir):
        if path != '.DS_Store':
            i += 1

            test_path = os.path.join('%s/%s' % (test_dir, path))
            test_data = get_data(test_path)
            test_x, test_y = test_data[:, 0:-1], test_data[:, -1]
            orig_acc, fpr = test(test_x, test_y)
            print path
            print('the acc is %.2f, the fpr is %.2f' % (orig_acc, fpr))

            #name = 't14' + str(i) + ' ' + 'roc_auc=' + str(round(roc_auc, 2))
            #name_lis.append(name)
            #plt.step(recall, precision, alpha=0.2,where = 'post')
            #plt.fill_between(recall, precision, step='post', alpha=0.2)
            #plt.ylim([0.0, 1.05])
            #plt.xlim([0.0, 1.0])

            #plt.plot(fpr, tpr, lw=1)

    #plt.xlabel('False positive rate')
    #plt.ylabel('True positive rate')
    #plt.xlabel('Recall')
    #plt.ylabel('Precision')
    #plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    #name_lis.append('diagonal')
    #plt.legend(name_lis, loc=0, ncol=2)
    #plt.title('2-class Precision-Recall curve')

    #plt.show()
    #plt.savefig('')

def find_forged():
    #这两个都不可以
    #modify0 = -0.3 * test_x0 - 0.8
    #modify1 = -0.9 * test_x1 + 0.6
    test_data11 = get_data(test_file11)

    test_data1 = get_data(test_file1)
    test_x1, test_y1 = test_data1[:, 0:-1], test_data1[:, -1]
    mean1 = np.mean(test_x1, axis=0)

    mean11 = np.mean(test_data11, axis=0)

    #the modify acc1 is 0.1550 modify1 = -0.9 * test_x1 + 0.3
    modify1 = -0.96 * test_x1
    modify1[:, 2] = -modify1[:, 2]
    modify1[:, 3] = -modify1[:, 3]
    t1 = modify1[:, 1].tolist()
    t2 = modify1[:, 2].tolist()
    modify1[:, 2] = t1
    modify1[:, 1] = t2

    #modify1[:, 1] = t3 * 4
    #modif_mean0 = np.mean(modify0, axis=0)
    modif_mean1 = np.mean(modify1, axis=0)

    print('the orignal mean is ')
    print(mean1)
    print('the modi7 mean is')
    print(mean11)
    print('the modify mean is ')
    #print(modif_mean0)
    print(modif_mean1)




find_forged()


#finaltest(test_dir)

#find original is 1, but pre is 0