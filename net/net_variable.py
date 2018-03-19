# -*- coding: utf-8 -*-
import tensorflow as tf
in_dim = 4
out_dim = 2

weights = {
    'w1': tf.Variable(tf.random_normal([in_dim, 16])),
    'w2': tf.Variable(tf.random_normal([16, 8])),
    'w3': tf.Variable(tf.random_normal([8, 8])),
    'out': tf.Variable(tf.random_normal([8, 2]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([16])),
    'b2': tf.Variable(tf.random_normal([8])),
    'b3': tf.Variable(tf.random_normal([8])),
    'out': tf.Variable(tf.random_normal([2]))
}

x = tf.placeholder('float', [None, in_dim])
y = tf.placeholder('float', [None, out_dim])
keep_prob = tf.placeholder('float')
