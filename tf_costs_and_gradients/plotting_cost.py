from __future__ import print_function

import tensorflow as tf
from matplotlib import pyplot as plt

# Graph Input
X = [1., 2., 3.]
Y = [2., 4., 6.]
n_samples = len(X)

# model weight
W = tf.placeholder(tf.float32)

# Construct a linear model
hypothesis = tf.multiply(X, W)

# Cost function
# for each y,
# c = sum(y' - y)^2
# take mean value
# c / n_samples
cost = tf.reduce_sum(tf.pow(hypothesis - Y, 2)) / n_samples

init = tf.global_variables_initializer()

# for graphs
W_val = []
cost_val = []

# Launch the graphs
sess = tf.Session()
sess.run(init)

for i in range(-30, 50):
    print(i * -0.1, sess.run(cost, feed_dict={W: i * 0.1}))
    W_val.append(i * 0.1)
    cost_val.append(sess.run(cost, feed_dict={W: i * 0.1}))

plt.plot(W_val, cost_val, 'ro')
plt.ylabel('cost')
plt.xlabel('W')
plt.show()
