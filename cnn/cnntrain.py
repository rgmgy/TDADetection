# -*- coding: utf-8 -*-
from getdata import get_traindata
import tensorflow as tf
from cnn import conv_net
from variable import weights, biases
import numpy as np

model_path = 'car-plane1.0.ckpt'
#train data
file_dir = '/Users/gy/Desktop/TDAML/transport/traincp'
data_x, data_y = get_traindata(file_dir)

'''
temp_x = []
temp_y = []
for i in range(3):
    s = (2 * i + 1) * 96
    e = (2 * i + 2) * 96
    for term in data_x[s:e, :]:
        temp_x.append(term)
    for term in data_y[s:e, :]:
        temp_y.append(term)
'''
train_x = data_x
train_y = data_y

#va_x = data_x[540:580, :]
#va_y = data_y[540:580, :]
print(train_x.shape, train_y.shape)

learning_rate = 0.001
batch_size = 100
dropout = 0.85
training_iters = 200
display_step = 2

n_input = 49152
n_classes = 2
x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_classes])
keep_prob = tf.placeholder('float')



pred, conv1_temp, conv2_temp = conv_net(x, weights, biases, keep_prob)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimmizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.name_scope('train'):
    with tf.name_scope('weights'):
        tf.histogram_summary('wc1', weights['wc1'])
        tf.histogram_summary('wc2', weights['wc2'])
        tf.histogram_summary('wd1', weights['wd1'])
        tf.histogram_summary('out', weights['out'])
        tf.histogram_summary('bc1', biases['bc1'])
        tf.histogram_summary('bc2', biases['bc2'])
        tf.histogram_summary('bd1', biases['bd1'])
        tf.histogram_summary('bout', biases['out'])

    with tf.name_scope('outcome'):
        tf.scalar_summary('loss', cost)
        tf.scalar_summary('acc', accuracy)




init = tf.initialize_all_variables()
saver = tf.train.Saver()
merged = tf.merge_all_summaries()



with tf.Session() as sess:
    train_writer = tf.train.SummaryWriter('/Users/gy/Desktop/summary' + '/train', sess.graph)
    sess.run(init)

    #total_batch = int(train_x.shape[0] / batch_size)
    va_acc = 0.
    va_n = 0
    for epoch in range(training_iters):
        sess.run(optimmizer, feed_dict={
            x: train_x, y: train_y, keep_prob: dropout
        })

        if epoch % display_step == 0 or epoch == 399:
            # predicted, conv1, conv2 = sess.run([pred, conv1_temp, conv2_temp],feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            summary, loss, acc = sess.run([merged, cost, accuracy], feed_dict={
                x: train_x, y: train_y, keep_prob: dropout
            })
            train_writer.add_summary(summary, epoch)
            print('epoch is %d' % epoch)
            print("the loss is %f" % loss)
            print("the acc is %f" % acc)


        #for i in range(total_batch):
            #batch_x = train_x[i * batch_size: (i + 1) * batch_size]
            #batch_y = train_y[i * batch_size: (i + 1) * batch_size]




    saver.save(sess, model_path)