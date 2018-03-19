# -*- coding: utf-8 -*-
from __future__ import print_function


import tensorflow as tf
model_path = 'transport1.1.ckpt'


def conv2d(x, W, b, strides=2):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')

    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(x , weights, biases, keep_prob):
    x = tf.reshape(x, [-1, 128, 128, 3])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])


    conv1 = maxpool2d(conv1)

    conv1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
              name='norm1')


    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2)

    #conv3 = maxpool2d(conv3)

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    #???????? about keep_prob
    fc1 = tf.nn.dropout(fc1, keep_prob)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out, conv1, conv2



