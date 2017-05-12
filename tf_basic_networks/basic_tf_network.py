# Basics NeuralNetwork of Tensorflow
import tensorflow as tf
import numpy as np
x_data = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
y_data = tf.constant([2.0, 4.0, 6.0, 8.0, 10.0])

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

weight = tf.Variable([1.0])
bias = tf.Variable([0.0])

init = tf.initialize_all_variables()
predict = (x * weight) 
loss = tf.reduce_mean(tf.square(predict - y) ) 
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss/2)
steps = 300
with tf.Session() as sess:
    sess.run(init)
    X = sess.run(x_data)
    Y = sess.run(y_data)
    for i in range(steps):
    	print 'Step: ', i, ':'
    	print 'Loss: ', sess.run(loss, {x: X, y: Y})
    	sess.run(optimizer, {x: X, y: Y})
    print 'THE FINAL WEIGHT IS', sess.run(weight)[0]

