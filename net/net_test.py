# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from net import network
from net_variable import weights, biases, x, y, keep_prob
from handledata.get_data import get_mldata
import os
from svm.ROC_SVM import Roc
import matplotlib.pyplot as plt

file_path = '/Users/gy/Documents/learn/maching_learning/mynote/mapper out/data/forpaper/banknote/banknote0/banknote0test.csv'
model_path = '/Users/gy/Documents/learn/maching_learning/mynote/mapper out/data/forpaper/banknote/train/bankmodel_shulle_final_ Oct'
test_dir = '/Users/gy/Desktop/paperPicture/combine/combine2/modify1'
#test_x, test_y = get_mldata(file_path)


pred = network(x, weights, biases, keep_prob)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, model_path)
    name_lis = []
    i = 0
    for path in os.listdir(test_dir):
        if path != '.DS_Store':
            i +=1
            test_path = os.path.join('%s/%s' % (test_dir, path))
            test_x, test_y, or_y = get_mldata(test_path)
            orignal_pred, orignal_acc, orignal_c = sess.run([pred, accuracy, cost],
                                                            feed_dict={x: test_x, y: test_y, keep_prob: 1.0})

            fpr, tpr, roc_auc, precision, recall = Roc(or_y, orignal_pred[:, 1])
            name = 't11' + str(i) + ' ' + 'roc_auc=' + str(round(roc_auc, 2))
            name_lis.append(name)
            plt.plot(fpr, tpr, lw=1)

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    name_lis.append('diagonal')
    plt.legend(name_lis, loc=0, ncol=2)

    plt.show()







