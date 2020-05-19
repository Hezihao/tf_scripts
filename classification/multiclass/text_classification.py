#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# download dataset
train_data, validation_data, test_data = tfds.load(
	name='imdb_reviews',
	split=('train[:60%]', 'train[60%:]', 'test'),
	as_supervised=True)

# exploring the data
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
#print(train_examples_batch)
#print(train_labels_batch)

# build the model
# pretrained text embedding model, to map sentences into structured vectors
# resulting dimensions: (num_examples, embedding_dimension)
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
						   dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

# build up whole network
model = tf.keras.Sequential()
model.add(hub_layer)
# the hidden layer(but still fully connected) with 16 neurons
model.add(tf.keras.layers.Dense(16, activation='relu'))
# the output layer with 1 single neuron
model.add(tf.keras.layers.Dense(1))

model.summary()			# printing model structure out, I guess it's useful when the structure is not so complicated

model.compile(optimizer='adam',
			  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),		# transform the loss value from log function on [-Inf, +Inf] to a value on [0, 1]
			  metrics=['accuracy'])												# cause in the learning process, it needs logit to describe the error on whole R.
print(train_data.shape())
history = model.fit(train_data.shuffle(10000).batch(512),
					epochs=28,
					validation_data=validation_data.batch(512),
					verbose=1)

results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
	print("%s: %.3f" % (name, value))