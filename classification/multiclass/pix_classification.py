#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras

# load dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# preprocessing
train_images = train_images / 255
test_images = test_images / 255

# labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']	# a list for look up name with a code
print(train_images.shape)
print(train_labels)

# model
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
	keras.layers.Dense(128, activation='relu'),		# basically irrelevant number of neurons, just deciding how complex the layer is
	keras.layers.Dense(10)							# 10 = nums of classes, it's the output, can I say it's something similar to a decoder?
])
model.compile(optimizer='adam',
			  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),		# learn about Entropy, it's a widely used loss function.
			  metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)

# make predictions
probability_model = tf.keras.Sequential([model,
										tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# visualization
plt.figure('Example')
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)

plt.figure(figsize=(10, 10))
for i in range(25):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(train_images[i], cmap=plt.cm.binary)
	plt.xlabel(class_names[train_labels[i]])
	print(type(class_names[train_labels[i]]))
plt.show()