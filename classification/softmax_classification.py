from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

filepath = os.path.abspath(__file__)
PWDPATH = os.path.dirname(filepath)
DATAPATH = PWDPATH + '/softmax_train.txt'

xy = np.loadtxt(DATAPATH, unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])


X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])

W = tf.Variable(tf.zeros([3, 3]))

# matrix shape X=[8, 3], W=[3, 3]
hypothesis = tf.nn.softmax(tf.matmul(X, W))

learning_rate = 0.001

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
