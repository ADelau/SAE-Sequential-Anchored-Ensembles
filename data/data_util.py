import os
import numpy as np
import tensorflow as tf

def load_data(dataset_name, train_val_split=False, normalize=False, seed=None):
	""" Load the datasets

	Args:
		dataset_name (str): The name of the dataset to load
		train_val_split (bool, optional): Set to True to split the train set in a train and validation sets. Defaults to False.
		normalize (bool, optional): Set to True to normalize the data. Defaults to False.
		seed (int, optional): The seed to use for the train/val split. Defaults to None (unseeded).

	Returns:
		tuple: (train_set, val_set, test_set, probas)
			train_set (tf.data.Dataset): the training set
			val_set (tf.data.Dataset): the validation set (return None if train_val_split is set to False)
			test_set (tf.data.Dataset): the test set
			probas (np.array): The HMC prediction probabilities (return None if does not exist)
	"""

	datadir = os.path.join("data", dataset_name)

	# Load Cifar10 dataset
	if dataset_name == "cifar10":
		train_x = np.loadtxt(os.path.join(datadir, "{}_train_x.csv".format(dataset_name)))
		train_y = np.loadtxt(os.path.join(datadir, "{}_train_y.csv".format(dataset_name)))
		test_x = np.loadtxt(os.path.join(datadir, "{}_test_x.csv".format(dataset_name)))
		test_y = np.loadtxt(os.path.join(datadir, "{}_test_y.csv".format(dataset_name)))

		# Load the HMC prediction probabilities if they exist
		try:
			probas = np.loadtxt(os.path.join(datadir, "probs.csv"))
		except OSError:
			probas = None

		# Reshape the images
		train_x = train_x.reshape((len(train_x), 32, 32, 3))
		test_x = test_x.reshape((len(test_x), 32, 32, 3))

		# Set the mean and std for normalization
		mean_ = tf.constant((0.49, 0.48, 0.44), shape=[1, 1, 1, 3], dtype=train_x.dtype)
		std_ = tf.constant((0.2, 0.2, 0.2), shape=[1, 1, 1, 3], dtype=train_x.dtype)

	# Load IMDB dataset
	if dataset_name == "imdb":
		train_x = np.loadtxt(os.path.join(datadir, "{}_train_x.csv".format(dataset_name))).astype(int)
		train_y = np.loadtxt(os.path.join(datadir, "{}_train_y.csv".format(dataset_name)))
		test_x = np.loadtxt(os.path.join(datadir, "{}_test_x.csv".format(dataset_name))).astype(int)
		test_y = np.loadtxt(os.path.join(datadir, "{}_test_y.csv".format(dataset_name)))
		
		# Load the HMC prediction probabilities if they exist
		try:
			probas = np.loadtxt(os.path.join(datadir, "probs.csv"))
		except OSError:
			probas = None

	# Load Cifar10 corrupted, dermaMNIST or UCI-GAP dataset
	if dataset_name == "cifar_anon" or dataset_name == "dermamnist_anon" or dataset_name == "energy_anon":

		# Load npz file
		data = np.load(os.path.join(datadir, "{}.npz".format(dataset_name)))

		# Retrieve the datasets
		train_x = data["x_train"]
		train_y = data["y_train"]
		test_x = data["x_test"]
		test_y = data["y_test"]

		# Load the HMC prediction probabilities if they exist
		try:
			probas = np.loadtxt(os.path.join(datadir, "probs.csv"))
		except OSError:
			probas = None

	# Set the normalization constants
	if normalize:
		if dataset_name == "cifar_anon" or dataset_name == "dermamnist_anon":

			mean_ = np.mean(train_x, axis=(0,1,2))
			std_ = np.std(train_x, axis=(0,1,2))
			print("mean = {}".format(mean_))
			print("std = {}".format(std_))

			mean_ = tf.constant(mean_, shape=[1, 1, 1, 3], dtype=train_x.dtype)
			std_ = tf.constant(std_, shape=[1, 1, 1, 3], dtype=train_x.dtype)

			print("mean = {}".format(mean_))
			print("std = {}".format(std_))

		if dataset_name == "energy_anon":
			
			mean_ = np.mean(train_x, axis=0)
			std_ = np.std(train_x, axis=0)
			print("mean = {}".format(mean_))
			print("std = {}".format(std_))

			mean_ = tf.constant(mean_, shape=[8], dtype=train_x.dtype)
			std_ = tf.constant(std_, shape=[8], dtype=train_x.dtype)

			print("mean = {}".format(mean_))
			print("std = {}".format(std_))

	# Split the training set in train and validation set
	if train_val_split:
		train_len = len(train_x)
		indices = np.arange(train_len)

		# If a seed is specified, seed the random number generator
		if seed is not None:
			print("seeding split with seed {}".format(seed))
			np.random.seed(seed)

		# Select the indices for the train and validation sets
		np.random.shuffle(indices)
		train_indices = indices[:int(train_len*0.9)]
		val_indices = indices[int(train_len*0.9):]
		
		# Build the train and validations sets
		val_x = train_x[val_indices]
		val_y = train_y[val_indices]
		train_x = train_x[train_indices]
		train_y = train_y[train_indices]

	# Normalize the data
	if normalize:
		train_x = (train_x - mean_)/std_
		test_x = (test_x - mean_)/std_
		if train_val_split:
			val_x = (val_x - mean_)/std_

	# Cast the images in float for the dermaMNIST dataset
	elif dataset_name == "dermamnist_anon":
		train_x = train_x.astype(np.float32)
		if train_val_split:
			val_x = val_x.astype(np.float32)
		test_x = test_x.astype(np.float32)

	# Construct the tf datasets
	train_set = tf.data.Dataset.from_tensor_slices((train_x, train_y))
	test_set = tf.data.Dataset.from_tensor_slices((test_x, test_y))

	if train_val_split:
		val_set = tf.data.Dataset.from_tensor_slices((val_x, val_y))

	else:
		val_set = None
	
	return train_set, val_set, test_set, probas
