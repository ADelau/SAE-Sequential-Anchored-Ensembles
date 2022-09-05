Config files take the form of Yaml files. Each field represents an argument for the program and its associated value.

## Base arguments
These arguments should always be provided.

| Name | Description |
| --- | --- |
| dataset_name | The name of the dataset: cifar10, imdb, cifar_anon, dermamnist_anon, energy_anon or a dataset ou have added |
| method |  |
| train_batch_size |  |
| test_batch_size |  |
| nb_epochs |  |
| optimizer |  |
| lr |  |
| min_lr |  |
| save_dir |  |
| keep_best_weights |  |
| train_val_split |  |
| competition_mode |  |

Optionally, the following arguments can be provided.
| Name | Description |
| --- | --- |
| seed |  |
| save_anchors |  |
| max_budget |  |
| early_stopping |  |
| early_stopping_epochs |  |

## Method arguments
Some of these arguments should be provided depending on the method used. 

* For the simple model, no additional arguments are required. 
* Ensembles:
  | Name | Description |
  | --- | --- |
  | ensemble_size |  |
  | anchored | Optionnal, defaults to True |
  
* Sequential ensembles:
  | Name | Description |
  | --- | --- |
  | ensemble_size |  |
  | num_chains |  |
  | sequential_nb_epochs |  |
  | sequential_optimizer |  |
  | sequential_lr |  |
  | sequential_min_lr |  |
  | sampler |  |

* Graphical ensembles:
  | Name | Description |
  | --- | --- |
  | ensemble_size |  |
  | sequential_nb_epochs |  |
  | sequential_optimizer |  |
  | sequential_lr |  |
  | sequential_min_lr |  |
  | distance |  |
  | num_init_point |  |

## Optimizer arguments
Depending on the optimizer used, some additionnel arguments should be provided.

* SGD
  | Name | Description |
  | --- | --- |
  | momentum |  |
  | nesterov |  |
* Adam
  | Name | Description |
  | --- | --- |
  | b1 |  |
  | b2 |  |

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
