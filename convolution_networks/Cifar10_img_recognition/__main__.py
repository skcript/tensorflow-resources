import tensorflow as tf
import numpy as np
from conf import *
from cifar10 import *

# Working of our algorithm is as follows:
# Conv1_layer -> Conv2_layer -> Flatten_layer -> FullyConnected_layer -> FullyConnected_layer(With 10 Classes)

# Reading the data
image_data, image_cls, img_one_hot_cls = (load_training_data())

image_data_flat = image_data.reshape([-1,3072])

# Function for defining weights
def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

# Function for defining biases
def new_bias(length):
	return tf.Variable(tf.constant(0.5, shape=[length]))

# Function to create the convolution layer with/without max-pooling
def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
	shape = [filter_size, filter_size, num_input_channels, num_filters]
	weights = new_weights(shape = shape)
	biases = new_bias(length = num_filters)

	# tf.nn.conv2d needs a 4D input
	layer = tf.nn.conv2d(input = input, filter= weights, strides=[1,1,1,1], padding='SAME')
	layer += biases
	if use_pooling:
		layer = tf.nn.max_pool(value = layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	# relu activation function converts all negatives to zero
	layer = tf.nn.relu(layer)
	return layer, weights

# After all convolutions, we need to flatten the layer
def flatten_layer(layer):
	layer_shape = layer.get_shape()
	num_features = layer_shape[1:4].num_elements()
	layer_flat = tf.reshape(layer, [-1, num_features])
	return layer_flat, num_features

# Fully connected layer
def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
	weights = new_weights(shape=[num_inputs, num_outputs])
	biases = new_bias(length= num_outputs)
	layer = tf.matmul(input, weights) + biases
	if use_relu:
		layer = tf.nn.relu(layer)
	return layer


# The placeholder to hold the X and Y values while training
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true_cls = tf.argmax(y_true, dimension=1)

# The beginning of the process
layer_conv1, weights_conv1 = new_conv_layer(input = x_image, num_input_channels= num_channels, filter_size = filter1_size, num_filters = number_of_filter1, use_pooling=True)
layer_conv2, weights_conv2 = new_conv_layer(input = layer_conv1, num_input_channels= number_of_filter1, filter_size = filter2_size, num_filters = number_of_filter2, use_pooling=True)
layer_flat, num_features = flatten_layer(layer_conv2)
layer_fc1 = new_fc_layer(layer_flat, num_features, fc_size, True)
layer_fc2 = new_fc_layer(layer_fc1, fc_size, num_classes, False)

# Finally Softmax function
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)

# Cost function calculation and optimization function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# Checking for the right predictions
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TF Session initiation
session = tf.Session()
session.run(tf.global_variables_initializer())

# The trainer function to iterate the training process to learn further
def optimize(num_iterations):
	x_batch, y_true_batch = image_data_flat, img_one_hot_cls
	feed_dict_train = {x: x_batch[:500], y_true: y_true_batch[:500]}
	feed_dict_test = {x: x_batch[500:1000], y_true: y_true_batch[500:1000]}
	for i in range(num_iterations):
		session.run(optimizer, feed_dict=feed_dict_train)
		# Print status every 10 iterations.
		if i % 10 == 0:
		# Calculate the accuracy on the training-set.
			acc = session.run(accuracy, feed_dict=feed_dict_test)

		# Message for printing.
			print "Step ",i+1,': ', acc*100

optimize(STEPS)

# test_data = image_data[1115].reshape([1,3072])
# print image_cls[1115]
# print session.run(y_pred_cls, {x: test_data})
