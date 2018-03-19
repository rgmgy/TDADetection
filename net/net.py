# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


def Layer(x, weights, basies, keep_prob):

    out = tf.matmul(x, weights) + basies
    out = tf.nn.relu(out)
    out = tf.nn.dropout(out, keep_prob)
    return out

def network(x, weights, basies, keep_prob):

    l1 = Layer(x, weights['w1'], basies['b1'], keep_prob)
    l2 = Layer(l1, weights['w2'], basies['b2'], keep_prob)
    #l3 = Layer(l2, weights['w3'], basies['b3'], keep_prob)


    out = tf.add(tf.matmul(l2, weights['out']), basies['out'])
    out = tf.nn.softmax(out)
    return out

