# Tensorflow Basics on NeuralNetwork with hidden layer

import tensorflow as tf
import numpy as np

INPUT_NEURONS = 2
HIDDEN_LAYER_1_NEURONS = 4
OUTPUT_NEURONS = 1
x_data = np.array([[0, 0], [0, 1], [1, 1]])

y_data = np.array([0, 1, 1])

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

h1_layer = {'weight': tf.Variable(np.random.rand(INPUT_NEURONS, HIDDEN_LAYER_1_NEURONS).astype(np.float32)),
            'bias': tf.constant(0.0)}
output_layer = {'weight': tf.Variable(np.random.rand(HIDDEN_LAYER_1_NEURONS, OUTPUT_NEURONS).astype(np.float32)),
                'bias': tf.constant(0.0)}


h1 = tf.matmul(x, h1_layer['weight']) + h1_layer['bias']
h1 = tf.sigmoid(h1)

predict = tf.matmul(h1, output_layer['weight']) + output_layer['bias']

loss = tf.reduce_mean(tf.square(predict - y))
optimizer = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(50000):
		if i % 5000 == 0:
			print 'Loss: ', sess.run(loss, {x: x_data, y: y_data})
		sess.run(optimizer, {x: x_data, y: y_data})

print sess.run(predict, {x: [[1, 1]] })
