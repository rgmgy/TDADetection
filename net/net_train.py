# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from net import network
from net_variable import weights, biases, x, y, keep_prob
from handledata.get_data import get_mldata

file_path = '/Users/gy/Documents/learn/maching_learning/mynote/mapper out/data/forpaper/banknote/train/banknotetrain.csv'
model_path = '/Users/gy/Documents/learn/maching_learning/mynote/mapper out/data/forpaper/banknote/train/bankmodel_shulle_final_ Oct'
train_x, train_y = get_mldata(file_path, True)
#现在可以用的model4。model5效果更好



learning_rate = 0.0001
iter_time = 5000
display_time = 50
dropout = 1.0


pred = network(x, weights, biases, keep_prob)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimmizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    acc = 0.
    for epoch in range(iter_time):
        sess.run(optimmizer, feed_dict={x: train_x, y:train_y, keep_prob: dropout})
        if epoch % display_time == 0 or epoch == iter_time-1:
            acc, c = sess.run([accuracy, cost], feed_dict={x: train_x, y:train_y, keep_prob: dropout})

            print('the epoch %d, the cost %.4f, the acc %.4f' % (epoch, c, acc))
        if acc > 0.97:
            break

    saver.save(sess, model_path)