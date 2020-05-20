#!/usr/bin/env python3

import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras

(train_imgs, train_labels), (test_imgs, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

plt.figure(1)
plt.subplot(2,2,1)
plt.imshow(train_imgs[10])

train_imgs = train_imgs[:1000].reshape(-1, 28 * 28) / 255.0		# flat the image into a vector of 784 dimensions
print(train_imgs[10])

plt.show()
