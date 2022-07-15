The five following datasets are available for use:
* Cifar10 (cifar10, development phase)
* IMDB (imdb, development phase)
* Cifar10 corrupted (cifar_anon, evaluation phase)
* MedMNIST (dermamnist_anon, evaluation phase)
* UCI-Gap (energy_anon, evaluation phase)

For all those datasets, the following are provided:
* Training set features and labels
* Test set features, labels and HMC predictions

The features and labels can be found on the [competition website](https://izmailovpavel.github.io/neurips_bdl_competition/getting_started.html).

HMC predictions have been computed by the competition organizers using [this codebase](https://github.com/google-research/google-research/tree/master/bnn_hmc). The predictions for the Cifar10 and IMDB datasets are available publicly [here](https://github.com/izmailovpavel/neurips_bdl_starter_kit/tree/main/data). Simply copy the `probs.csv` files in the corresponding folders. The predictions for the Cifar10 corrupted, MedMNIST and UCI-Gap have been shared privatly. Please, contact the competition organizers if you wish to have access to those. 

Features, labels and HMC predictions should be placed in the corresponding folders without renaming them.

Note 1: the neural network architectures are automatically selected to match the one used to generate the HMC predictions.

Note 2: The code will run fine without the `probs.csv` file, some metrics will simply not be computed.
