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
plt.figure(1)
for features, label in packed_ds.batch(1000).take(3):	# dataset.take(x) means take the first x elements from the dataset lying before the function.
	print(features[0])
	plt.hist(features.numpy().flatten(), bins = 101)
plt.show()

# setup training/test set for the experiment
N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE