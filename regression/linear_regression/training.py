#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# parameters of learning
num_epochs = 15000
rate = [0.01, 0.01]

# data preparation
X = np.add(np.array(range(100))*0.01, np.random.rand(100)*3)
Y = np.add(np.add(np.multiply(X, 5.0), 3.0), np.random.rand(100)*8)

# analysis
global error_list
error_list = []

# model setup
class LinearModel():
	def __init__(self):
		# important ! Initialize variables as float type
		self.Weight = tf.Variable(1.0)
		self.Bias = tf.Variable(1.0)
	def __call__(self, x):
		return self.Weight * x + self.Bias

def loss(y, pred):
	global current_loss
	global error_list
	# RMSE
	current_loss = tf.sqrt(tf.reduce_mean(tf.square(y - pred)))
	error_list.append(current_loss)
	return current_loss

my_model = LinearModel()
M = []
b = []
for epoch in range(num_epochs):
	M.append(my_model.Weight.numpy())
	b.append(my_model.Bias.numpy())
	with tf.GradientTape() as t:
		step_loss = loss(Y, my_model(X))
	step_weight, step_bias = t.gradient(step_loss, [my_model.Weight, my_model.Bias])
	my_model.Weight.assign_sub(step_weight * rate[0])
	my_model.Bias.assign_sub(step_bias * rate[1])
	print("Epoch: "+str(epoch)+" Error: "+str(current_loss.numpy()))

# visualization
plt.figure(1)
plt.plot(X, Y, 'rx', label='Original data')
plt.plot(X, my_model(X), 'b-', label='Learned line')
plt.legend()

plt.figure(2)
plt.plot(range(len(error_list)), error_list, label='Error history')
plt.legend()

plt.figure(3)
plt.plot(range(len(M)), M, label='Weight history')
plt.plot(range(len(b)), b, label='Bias history')
plt.legend()

plt.show()