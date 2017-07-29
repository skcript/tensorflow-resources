import tensorflow as tf
import numpy as np
import os

from scipy.ndimage import *
from config import *


def load_images():
  data = {"images":[], "labels":[]}
  horses = [file for file in os.listdir(RESIZED_IMAGE_FOLDER) if file.startswith("horse") ]
  dogs = [file for file in os.listdir(RESIZED_IMAGE_FOLDER) if file.startswith("dog") ]
  for horse in horses:
    image = imread(os.path.join(RESIZED_IMAGE_FOLDER, horse), mode='L')
    data['images'].append(image)
    data['labels'].append([1,0])

  for dog in dogs:
    image = imread(os.path.join(RESIZED_IMAGE_FOLDER, dog), mode='L')
    data['images'].append(image)
    data['labels'].append([0,1])
  return data

def new_weights(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
 
def new_biases(length):
  return tf.Variable(tf.constant(0.05, shape=[length]))
 
def convolution_layer(input, num_of_input_channels, filter_size, num_of_filters, pooling=True):
  shape = [filter_size, filter_size,  num_of_input_channels, num_of_filters]
  weights = new_weights(shape=shape)
  biases = new_biases(length=num_of_filters)
  layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
  layer += biases

  if pooling:
      layer = max_pooling_layer(layer)
  layer = tf.nn.relu(layer)
  return layer, weights

def max_pooling_layer(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
  return tf.nn.max_pool(layer, ksize, strides, padding)


def flatten_layer(layer):
  layer_shape = layer.get_shape()
  num_features = layer_shape[1:4].num_elements()

  layer_flat = tf.reshape(layer, [-1, num_features])
  return layer_flat, num_features
 
def fully_connected_layer(input, num_inputs, num_outputs, relu=True): 
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
 
    layer = tf.matmul(input, weights) + biases
    if relu:
        layer = tf.nn.relu(layer)
 
    return layer



data = load_images()
images, labels = data['images'], data['labels']
classes = ['horse', 'dog']
num_classes = 2
img_size = 128
num_channels = 1
filter_size = 3
num_filters = 1
epochs = 100000

img_size_flat= img_size * img_size * num_channels
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


""" 
The Model
"""
layer_conv, weights_conv = convolution_layer(x_image, num_channels, filter_size, num_filters, pooling=True)
layer_flat, num_features = flatten_layer(layer_conv)
layer_fc = fully_connected_layer(layer_flat, num_features, num_classes, relu=True)

y_pred_cls = tf.argmax(layer_fc, 1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc, labels=y_true)
cost = tf.reduce_sum(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

""" The Process """
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(epochs):
    if(i % 10000) == 0:
      print '\nEpoch {0}'.format(i)
      sess.run(optimizer, {x: np.array(images).reshape(-1, 128*128),  y_true:np.array(labels)})
      print "Loss is {0}".format(sess.run(cost, {x: np.array(images).reshape(-1, 128*128),  y_true:np.array(labels)} ))

      result = sess.run(correct_prediction, {x: np.array(images).reshape(-1, 128*128),  y_true:np.array(labels)} )
      print "The number of right outputs: {0} out of {1}".format(sum(result), len(result))