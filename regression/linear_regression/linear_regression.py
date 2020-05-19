#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

learning_rate = 0.01
training_epochs = 1000
display_step = 50

# ground truth, used to make up the dataset
W0 = np.add(np.dot(np.random.rand(100),3), 5)
B0 = np.add(np.dot(np.random.rand(100),3), 3)

train_X = np.array(range(100))
train_Y = np.add(np.multiply(W0, train_X), B0)

n_samples = train_X.shape[0]

def loss(y, pred):
	# MAE
	# return tf.abs(tf.reduce_sum(y - pred))/y.shape[0]
	# MSE (will not converge)
	# return tf.reduce_mean(tf.square(y - pred))
	# RMSE
	# return tf.sqrt(tf.reduce_mean(tf.square(y - pred)))
	# R Squared (MSE/variance)
	E = 1.0 - (tf.reduce_mean(tf.square(y - pred)))/tf.cast(tf.reduce_mean(tf.square(y - tf.reduce_mean(y))), tf.float32)
	return tf.sqrt(tf.reduce_mean(tf.square(y - pred)))

# X = tf.placeholder("float")
# Y = tf.placeholder("float")
# W = tf.Variable(np.random.randn(), name="weight")
# b = tf.Variable(np.random.randn(), name="bias")
class LinearModel:
	def __call__(self, x):
		return self.Weight * x + self.Bias

	def __init__(self):
		self.Weight = tf.Variable(10.0)
		self.Bias = tf.Variable(12.0)

linear_model = LinearModel()
W, b = [], []
for epoch_cnt in range(training_epochs):
	W.append(linear_model.Weight.numpy())
	b.append(linear_model.Bias.numpy())
	# This model is not so linear, it converges better on RMSE
	cost = loss(train_Y, linear_model(train_X))

	with tf.GradientTape() as t:
		current_loss = loss(train_Y, linear_model(train_X))
	lr_weight, lr_bias = t.gradient(current_loss, [linear_model.Weight, linear_model.Bias])
	linear_model.Weight.assign_sub(learning_rate * lr_weight)
	linear_model.Bias.assign_sub(learning_rate * lr_bias)
	print("Epoch count: "+str(epoch_cnt)+" Loss value: "+str(cost.numpy()))



# prediction of model
# pred = tf.add(tf.multiply(X, W), b)
# Mean squared error
# cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)


# visualization
print(linear_model.Weight, linear_model.Bias)
plt.scatter(train_X, train_Y, label='Original data')
plt.plot(train_X, linear_model(train_X), 'r-', label='Fitted line')
#plt.plot(range(len(W)), W, label='Estimated weights')
#plt.plot(range(len(b)), b, label='Estimated biases')
plt.legend()
plt.show()