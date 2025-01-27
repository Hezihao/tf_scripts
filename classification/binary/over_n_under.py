#!/usr/bin/env python3

import shutil
import pathlib
import tempfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import tensorflow_docs.plots
import tensorflow_docs.modeling
import tensorflow_docs as tfdocs

from IPython import display
from tensorflow.keras import layers
from tensorflow.keras import regularizers

logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)
gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')
FEATURES = 28
dataset = tf.data.experimental.CsvDataset(gz, [float(),]*(FEATURES+1), compression_type="GZIP")

def pack_row(*row):
	label = row[0]
	features = tf.stack(row[1:], 1)
	return features, label

packed_ds = dataset.batch(10000).map(pack_row).unbatch()

# exploration the dataset
'''
plt.figure(1)
for features, label in packed_ds.batch(1000).take(3):	# dataset.take(x) means take the first x elements from the dataset lying before the function.
	print(features[0])
	plt.hist(features.numpy().flatten(), bins = 101)
plt.show()
'''

# setup training/test set for the experiment
N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()
validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)
print(train_ds)

# training procedure
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
	0.001,
	decay_steps=STEPS_PER_EPOCH*1000,
	decay_rate=1,
	staircase=False)

def get_optimizer():
	return tf.keras.optimizers.Adam(lr_schedule)

step = np.linspace(0, 100000)
lr = lr_schedule(step)
plt.figure(figsize = (8,6))
plt.plot(step/STEPS_PER_EPOCH, lr)		# create a smooth curve with discrete points
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('Learning Rate')

# generate TensorBoard logs for the training
def get_callbacks(name):
	return [tfdocs.modeling.EpochDots(),
			tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
			tf.keras.callbacks.TensorBoard(logdir/name),
	]

# compile and fit procedure
def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
	if optimizer is None:
		optimizer = get_optimizer()
	model.compile(optimizer=optimizer,
				  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
				  metrics=[
				  	tf.keras.losses.BinaryCrossentropy(
				  		from_logits=True, name='binary_crossentropy'),
				  		'accuracy'])
	model.summary()

	history = model.fit(
		train_ds,
		steps_per_epoch=STEPS_PER_EPOCH,
		epochs=max_epochs,
		validation_data=validate_ds,
		callbacks=get_callbacks(name),
		verbose=0)
	return history

# build models
tiny_model = tf.keras.Sequential([
	layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
	layers.Dense(1)
])

my_model = tf.keras.Sequential([
	layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
	layers.Dense(16, activation='elu'),
	layers.Dense(16, activation='elu'),
	layers.Dense(1)
])

my_model_regularized = tf.keras.Sequential([
	layers.Dense(16, activation='elu', kernel_regularizer=regularizers.l2(0.001),input_shape=(FEATURES,)),
	layers.Dense(16, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
	layers.Dense(16, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
	layers.Dense(1)
])

size_histories = {}
size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')
size_histories['My'] = compile_and_fit(my_model, 'My_design')
size_histories['My_regularized'] = compile_and_fit(my_model_regularized, 'My_regularized')

plotter = tfdocs.plots.HistoryPlotter(metric='binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])

# displaying it in tensorboard

plt.show()