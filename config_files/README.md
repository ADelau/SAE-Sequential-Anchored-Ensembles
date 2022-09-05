Config files take the form of Yaml files. Each field represents an argument for the program and its associated value.

## Base arguments
These arguments should always be provided.

| Name | Description |
| --- | --- |
| dataset_name | The name of the dataset: `cifar10`, `imdb`, `cifar_anon`, `dermamnist_anon`, `energy_anon` or a dataset you have added |
| method | The method to use: `simple_model`, `ensemble`, `sequential_ensemble` or `graph_ensemble` |
| train_batch_size | The batch size used for training |
| test_batch_size | The batch size used for testing |
| nb_epochs | The number of epochs to perform |
| optimizer | The optimizer to use: sgd or adam |
| lr | The initial learning rate |
| min_lr | The minimal learning rate atteined after scheduling (a cosine scheduling is applied). To turn of scheduling, simply set min_lr to the same value as lr.|
| save_dir | The name of the directory in which to save the results |
| keep_best_weights | Whether to keep the weights that achieved the best performance on the validation set (True) or the weights of the last epoch performed (False) |
| train_val_split | Whether to split the train test in train and validation sets: True or False |
| competition_mode | Whether to run the code in competition_mode (removes testing and logging): True or False |

Optionally, the following arguments can be provided.
| Name | Description |
| --- | --- |
| seed | The seed |
| save_anchors | Whether to save the different anchors used: True or False |
| max_budget | The maximal total budget (in epochs). If specified, stops the training after this budget is reached even if fewer models than specified have been trained. |
| early_stopping | Whether to stop the training when the performance on the validation set are not longer improving: True or False |
| early_stopping_epochs | The number of epochs with no improvement required before stopping the training |

## Method arguments
Some of these arguments should be provided depending on the method used. 

* For the simple model, no additional arguments are required. 
* Ensembles:
  | Name | Description |
  | --- | --- |
  | ensemble_size | The number of models to train |
  | anchored | If True, perform anchored ensembles, otherwise perform classical deep ensembles. Optionnal, defaults to True |
  
* Sequential ensembles:
  | Name | Description |
  | --- | --- |
  | ensemble_size | The number of models per chain to train |
  | num_chains | The number of chains to run. Each chain will perform a first initial training and (ensemble_size - 1) sequential trainings |
  | sequential_nb_epochs | The number of epochs to perform during sequential trainings |
  | sequential_optimizer | The optimizer to use for sequential trainings: sgd or adam |
  | sequential_lr | The initial learning rate to use for sequential training |
  | sequential_min_lr | The minimal learning rate atteined after scheduling for sequential trainings (a cosine scheduling is applied). To turn of scheduling, simply set min_lr to the same value as lr. |
  | sampler | The sampler used to sample the anchors: `iid`, `guided_walk`, `gaussian_mh` or `hmc` |

* Graphical ensembles:
  | Name | Description |
  | --- | --- |
  | ensemble_size | The number of models to train |
  | sequential_nb_epochs | The number of epochs to perform during sequential trainings |
  | sequential_optimizer | The optimizer to use for sequential trainings: sgd or adam |
  | sequential_lr | The initial learning rate to use for sequential training |
  | sequential_min_lr | The minimal learning rate atteined after scheduling for sequential trainings (a cosine scheduling is applied). To turn of scheduling, simply set min_lr to the same value as lr. |
  | distance | The distance function used to comparethe anchors: `l1`or `l2` |
  | num_init_point | The number of models to train fully before starting the sequential procedure. |

## Optimizer arguments
Depending on the optimizer used, some additionnel arguments should be provided.

* SGD
  | Name | Description |
  | --- | --- |
  | momentum | The momentum |
  | nesterov | Whether to used Nesterov momentum: True or False |
* Adam
  | Name | Description |
  | --- | --- |
  | b1 | Coefficient used for computing the running average of the gradient |
  | b2 | Coefficient used for computing the running average of the square of the gradient |

In cases a sequential optimizer is provided, those arguments should also be provided for the sequential optimizer (sequential_momentum, sequential_nesterov, sequential_b1, sequential_b2).

## Sampler arguments
When using the sequential ensembles method, depending on the sampler used, some additionnel arguments should be provided.

* iid sampler: No additional arguments
* gaussian_mh sampler
  | Name | Description |
  | --- | --- |
  | step_std |  |
* guided_walk sampler
  | Name | Description |
  | --- | --- |
  | step_std |  |
* hmc sampler
  | Name | Description |
  | --- | --- |
  | step_size |  |
  | num_steps |  |
