from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

filepath = os.path.abspath(__file__)
PWDPATH = os.path.dirname(filepath)
DATAPATH = PWDPATH + '/logistic_train.txt'

xy = np.loadtxt(DATAPATH, unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

h = tf.matmul(W, X)
hypothesis = tf.div(1., 1. + tf.exp(-h))

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

W_val = []
cost_val = []

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})

    W_val.append(sess.run(W)[0])
    cost_val.append(sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)[0])

fig, (ax) = plt.subplots(1, 1)
ax.set_ylim(0, 10)
ax.plot(W_val, cost_val)
plt.ylabel(sess.run(W))
plt.xlabel('W')
plt.show()
