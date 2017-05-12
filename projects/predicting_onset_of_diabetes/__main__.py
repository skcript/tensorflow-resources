import tensorflow as tf 
import pickle as pk
import numpy as np 
import pandas as pd
from conf import *

class NeuralNet:
	def __init__(self):
		self.declare_all_small_things()
		self.initialize_variables()

	def declare_all_small_things(self):
		# Reading and storing using Pandas
		x = pd.read_csv(FEATURES_CSV, header=None)
		x_data = x.values.astype(np.float32)
		y = pd.read_csv(OUTCOMES_CSV, header=None)
		y_data = y.values.astype(np.float32)

		# Splitting the train & test data
		self.x_train_data, self.x_test_data = x_data[:SPLIT], x_data[SPLIT:]
		self.y_train_data, self.y_test_data = y_data[:SPLIT], y_data[SPLIT:]

		# Some Constants Needed for the NN
		input_size = 8
		h1_size = 8
		h2_size = 8
		output_size = 2
		self.epochs = EPOCHS
		# End of the Constants

		# Our Network layers
		self.hidden_layer_1 = { 'weight': tf.Variable(tf.random_normal([input_size,h1_size])),
											 'bias': tf.Variable(tf.random_normal([1,h1_size])) }
		self.hidden_layer_2 = { 'weight': tf.Variable(tf.random_normal([h1_size,h2_size])), 
											 'bias': tf.Variable(tf.random_normal([1,h2_size])) }
		self.output_layer = { 'weight': tf.Variable(tf.random_normal([h2_size,output_size])), 
										 'bias': tf.Variable(tf.random_normal([1,output_size])) }

	def neural_networking(self,input):
		# The process
		h1 = tf.matmul(input, self.hidden_layer_1['weight']) + self.hidden_layer_1['bias']
		h1 = tf.sigmoid(h1)
		# h1 = tf.nn.relu(h1)
		h2 = tf.matmul(h1, self.hidden_layer_2['weight']) + self.hidden_layer_2['bias']
		# h2 = tf.sigmoid(h2)
		h2 = tf.nn.relu(h2)
		output = tf.matmul(h2, self.output_layer['weight']) + self.output_layer['bias']
		output = tf.nn.dropout(output, 1)
		return output

	def initialize_variables(self):
		# TensorFlow placeholder and variables 
		self.xs = tf.placeholder(tf.float32)
		self.ys = tf.placeholder(tf.float32)

		# Prediction 
		self.predict = self.neural_networking(self.xs)
		delta = tf.square(self.predict - self.ys)
		self.cost = tf.reduce_sum(delta)
		optimizer = tf.train.AdamOptimizer(0.01)
		self.prediction = optimizer.minimize(self.cost)

		#Creating Session
		self.sess = tf.Session()
		

	def train(self):
		self.sess.run(tf.initialize_all_variables())
		for i in range(self.epochs):
			if i%100 == 0:
				print "Step: ", i
				print 'LOSS: ', self.sess.run([self.cost], {self.xs:self.x_train_data, self.ys:self.y_train_data})
			self.sess.run(self.prediction, {self.xs:self.x_train_data, self.ys:self.y_train_data})

	def accuracy(self):
		#Accuracy Calculation
		op =  self.sess.run(self.predict, {self.xs: self.x_test_data})
		counter = 0
		num = 766 - SPLIT
		for i in range(num):
			if(np.argmax(self.y_test_data[i]) == np.argmax(op[i])):
				counter +=  1
		print 'Accuracy is :', (counter * 100) / (num)
		
	def save_model(self):
		saver = tf.train.Saver()
		save_path = saver.save(self.sess, MODEL_FILE)
		print save_path

	def load_model(self):
		saver = tf.train.Saver()
		self.sess = tf.Session()
		self.sess.run(tf.initialize_all_variables())
		saver.restore(self.sess, MODEL_FILE)

	def normalize_user_input(self, user_inputs):
		with open(PICKLE_FILE) as file:
			max_values = pk.load(file)
			min_values = pk.load(file)

		for i in range(len(user_inputs)):
			# Manual min-max scalar normalization
			user_inputs[i] = ( user_inputs[i] - min_values[i] ) / (max_values[i] - min_values[i])

		return user_inputs



# You can run like this if needed
# x = NeuralNet()
# x.load_model()
# x.accuracy()
# x.train()
# x.save_model()
# x.accuracy()



