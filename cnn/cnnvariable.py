# -*- coding: utf-8 -*-
import tensorflow as tf


weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 16])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 16, 16])),
    #'wc3': tf.Variable(tf.random_normal([5, 5, 64, 128])),
    'wd1': tf.Variable(tf.random_normal([8*8*16, 80])),
    'out': tf.Variable(tf.random_normal([80, 2]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([16])),
    'bc2': tf.Variable(tf.random_normal([16])),
    #'bc3': tf.Variable(tf.random_normal([128])),
    'bd1': tf.Variable(tf.random_normal([80])),
    'out': tf.Variable(tf.random_normal([2]))
}



