#!/usr/bin/env python3

import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# fetch the data
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset_path)

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
				'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values = "?", comment='\t', sep=" ", skipinitialspace=True)
# this is important before start doing anything to the dataset.
dataset = raw_dataset.copy()
#print(dataset.tail())

# Preprocessing
# Clean data, wipe all those NaNs out
print(dataset.isna().sum())
dataset = dataset.dropna()			# useful line, removing unknown values
# convert categorical value into one-hot
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
#print(dataset.tail())
print(dataset.head(10))
# split into train & test sets
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
# sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
# plt.show()
# split features from labels(the desired output of network)
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

def norm(x):
	return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# build the model
def build_model():
	model = keras.Sequential([
		layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
		layers.Dense(64, activation='relu'),
		layers.Dense(1)			# a single prediction of fuel
		])
	optimizer = tf.keras.optimizers.RMSprop(0.001)
	model.compile(loss='mse',
				  optimizer=optimizer,
				  metrics=['mae', 'mse'])
	return model

model = build_model()
model.summary()
# try the model out, but the result is definitely wrong, because it's not trained yet

'''
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)
'''

# train the model
EPOCHS = 1000

history = model.fit(
	normed_train_data, train_labels,
	epochs=EPOCHS, validation_split = 0.2, verbose=0,	# verbose = 0, not showing anything of training progress
	callbacks=[tfdocs.modeling.EpochDots()])
hist = pd.DataFrame(history.history)
hist['epoch']=history.epoch
print(hist.tail())

# visualization of training results
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plt.figure('MAE')
plotter.plot({'Basic': history}, metric='mae')
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')

plt.figure('MSE')
plotter.plot({'Basic': history}, metric='mse')
plt.ylim([0, 20])
plt.ylabel('MSE [MPG^2]')

# build a 2nd model
model2 = build_model()
# automatically stop training when the validation score doesn't improve
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
early_history = model2.fit(normed_train_data, train_labels,
						  epochs=EPOCHS, validation_split=0.2, verbose=0,
						  callbacks=[early_stop, tfdocs.modeling.EpochDots()])
plt.figure('Early Stopping')
plotter.plot({'Early Stopping': early_history}, metric='mae')
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')

#loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
#print("Loss of 1st model: "+str(loss)+" MAE of 1st model: "+str(mae)+" MSE of 1st model: "+str(mse))
#loss, mae, mse = model2.evaluate(normed_test_data, test_labels, verbose=2)
#print("Loss of 2nd model: "+str(loss)+" MAE of 2nd model: "+str(mae)+" MSE of 2nd model: "+str(mse))

predicted_labels = model2.predict(normed_test_data)
print(predicted_labels[-10:])
print(test_labels[-10:])
plt.show()