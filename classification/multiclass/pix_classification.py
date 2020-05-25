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
# visualize the 1st pic in examples
plt.figure('Example 1')
plt.imshow(train_images[0])			# used for examing those false detections here
plt.colorbar()
print(train_images.shape)
print(train_images[0].shape)
plt.xlabel(class_names[np.argmax(probability_model.predict(np.expand_dims(train_images[0], 0)))])		# expand the dimension of 1 sample, to get it fit the input format
plt.grid(False)

# visualize training dataset(first 25 pix)
plt.figure('predictions', figsize=(10, 10))
faults = []
for i in range(25):
	offset = 50
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	# plt.imshow(train_images[i], cmap=plt.cm.binary)
	# plt.xlabel(class_names[train_labels[i]])
	# print(type(class_names[train_labels[i]]))
	plt.imshow(test_images[i+offset], cmap=plt.cm.binary)
	plt.xlabel(class_names[np.argmax(predictions[i+offset])]+'/'+str(class_names[test_labels[i+offset]] == class_names[np.argmax(predictions[i+offset])]))
	if(not class_names[test_labels[i+offset]] == class_names[np.argmax(predictions[i+offset])]):
		faults.append(i+offset)
print(faults)
plt.show()