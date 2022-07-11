import os
import numpy as np
import tensorflow as tf

def load_data(dataset_name, train_val_split=False, normalize=False, seed=None):
	datadir = os.path.join("data", dataset_name)

	if dataset_name == "toy":
		train_x = np.loadtxt(os.path.join(datadir, "{}_train_x.csv".format(dataset_name)))
		train_y = np.loadtxt(os.path.join(datadir, "{}_train_y.csv".format(dataset_name)))
		test_x = np.loadtxt(os.path.join(datadir, "{}_test_x.csv".format(dataset_name)))
		test_y = np.loadtxt(os.path.join(datadir, "{}_test_y.csv".format(dataset_name)))
		probas = np.loadtxt(os.path.join(datadir, "probs.csv"))

	if dataset_name == "cifar10" or dataset_name == "cifar10test":
		train_x = np.loadtxt(os.path.join(datadir, "{}_train_x.csv".format(dataset_name)))
		train_y = np.loadtxt(os.path.join(datadir, "{}_train_y.csv".format(dataset_name)))
		test_x = np.loadtxt(os.path.join(datadir, "{}_test_x.csv".format(dataset_name)))
		test_y = np.loadtxt(os.path.join(datadir, "{}_test_y.csv".format(dataset_name)))
		probas = np.loadtxt(os.path.join(datadir, "probs.csv"))
	
		train_x = train_x.reshape((len(train_x), 32, 32, 3))
		test_x = test_x.reshape((len(test_x), 32, 32, 3))
		mean_ = tf.constant((0.49, 0.48, 0.44), shape=[1, 1, 1, 3], dtype=train_x.dtype)
		std_ = tf.constant((0.2, 0.2, 0.2), shape=[1, 1, 1, 3], dtype=train_x.dtype)

	if dataset_name == "imdb" or dataset_name == "imdbtest":
		train_x = np.loadtxt(os.path.join(datadir, "{}_train_x.csv".format(dataset_name))).astype(int)
		train_y = np.loadtxt(os.path.join(datadir, "{}_train_y.csv".format(dataset_name)))
		test_x = np.loadtxt(os.path.join(datadir, "{}_test_x.csv".format(dataset_name))).astype(int)
		test_y = np.loadtxt(os.path.join(datadir, "{}_test_y.csv".format(dataset_name)))
		probas = np.loadtxt(os.path.join(datadir, "probs.csv"))

	if dataset_name == "retinopathy" or dataset_name == "retinopathy_test":
		data = np.load(os.path.join(datadir, "data.npz"))
		train_x = data["x_train"]
		train_y = data["y_train"]
		test_x = data["x_test"]
		test_y = data["y_test"]
		probas = None

	if dataset_name == "cifar_anon" or dataset_name == "cifar_anon_test" or \
			dataset_name == "dermamnist_anon" or dataset_name == "dermamnist_anon_test" or \
			dataset_name == "energy_anon" or dataset_name == "energy_anon_test" or dataset_name == "energy_anon_1_1" or \
			dataset_name == "energy_anon_1_2" or dataset_name == "energy_anon_1_4" or dataset_name == "energy_anon_1_8" or \
			dataset_name == "energy_anon_1_15" or dataset_name == "energy_anon_1_30" or dataset_name == "energy_anon_1_60" or \
			dataset_name == "energy_anon_1_120" or dataset_name == "energy_anon_1_240" or dataset_name == "energy_anon_1_480":

		data = np.load(os.path.join(datadir, "data.npz"))
		train_x = data["x_train"]
		train_y = data["y_train"]
		test_x = data["x_test"]
		test_y = data["y_test"]
		probas = np.loadtxt(os.path.join(datadir, "probs.csv"))

	if normalize:
		if dataset_name == "retinopathy" or dataset_name == "retinopathy_test" or dataset_name == "cifar_anon" or dataset_name == "cifar_anon_test" or \
				dataset_name == "dermamnist_anon" or dataset_name == "dermamnist_anon_test":

			mean_ = np.mean(train_x, axis=(0,1,2))
			std_ = np.std(train_x, axis=(0,1,2))
			print("mean = {}".format(mean_))
			print("std = {}".format(std_))

			mean_ = tf.constant(mean_, shape=[1, 1, 1, 3], dtype=train_x.dtype)
			std_ = tf.constant(std_, shape=[1, 1, 1, 3], dtype=train_x.dtype)

			print("mean = {}".format(mean_))
			print("std = {}".format(std_))

		if dataset_name == "energy_anon" or dataset_name == "energy_anon_test" or dataset_name == "energy_anon_1_1" or \
			dataset_name == "energy_anon_1_2" or dataset_name == "energy_anon_1_4" or dataset_name == "energy_anon_1_8" or \
			dataset_name == "energy_anon_1_15" or dataset_name == "energy_anon_1_30" or dataset_name == "energy_anon_1_60" or \
			dataset_name == "energy_anon_1_120" or dataset_name == "energy_anon_1_240" or dataset_name == "energy_anon_1_480":
			
			mean_ = np.mean(train_x, axis=0)
			std_ = np.std(train_x, axis=0)
			print("mean = {}".format(mean_))
			print("std = {}".format(std_))

			mean_ = tf.constant(mean_, shape=[8], dtype=train_x.dtype)
			std_ = tf.constant(std_, shape=[8], dtype=train_x.dtype)

			print("mean = {}".format(mean_))
			print("std = {}".format(std_))

	if train_val_split:
		train_len = len(train_x)
		
		indices = np.arange(train_len)
		if seed is not None:
			print("seeding split with seed {}".format(seed))
			np.random.seed(seed)

		np.random.shuffle(indices)
		train_indices = indices[:int(train_len*0.9)]
		val_indices = indices[int(train_len*0.9):]
		
		val_x = train_x[val_indices]
		val_y = train_y[val_indices]
		train_x = train_x[train_indices]
		train_y = train_y[train_indices]

	if normalize:
		train_x = (train_x - mean_)/std_
		test_x = (test_x - mean_)/std_
		if train_val_split:
			val_x = (val_x - mean_)/std_

	elif dataset_name == "dermamnist_anon" or dataset_name == "dermamnist_anon_test":
		train_x = train_x.astype(np.float32)
		if train_val_split:
			val_x = val_x.astype(np.float32)
		test_x = test_x.astype(np.float32)

	"""
	if dataset_name == "cifar10" or dataset_name == "cifar10test" or dataset_name == "retinopathy" or dataset_name == "retinopathy_test":
		train_x = (train_x - mean_)/std_
		test_x = (test_x - mean_)/std_
		if train_val_split:
			val_x = (val_x - mean_)/std_


	if dataset_name == "dermamnist_anon" or dataset_name == "dermamnist_anon_test":
		train_x = train_x/255
		test_x = test_x/255
		if train_val_split:
			val_x = val_x/255
	"""

	train_set = tf.data.Dataset.from_tensor_slices((train_x, train_y))
	test_set = tf.data.Dataset.from_tensor_slices((test_x, test_y))

	if train_val_split:
		val_set = tf.data.Dataset.from_tensor_slices((val_x, val_y))

	else:
		val_set = None
	

	return train_set, val_set, test_set, probas
