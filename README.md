# SAE: Sequential Anchored Ensembles
This repository contains the code associated with the manuscript `SAE: Sequential Anchored Ensembles` [[link](https://arxiv.org/abs/2201.00649)]. 

## Setting up the environment
The code has been tested with the version 3.7.11 of python. The requirements can be found in the file `requirements.txt`. To install all the dependencies, run the following command.
```
pip install -r requirements.txt
```

## Downloading the data
Instructions for downloading the data can be found [here](data/).

## Executing the pipeline
The pipeline (training and evaluation) can be executed by running the command 
```
python main.py --config_file "config file path" --index "index"
```

The structure of a config file and examples of config files can be found [here](config_files/). The index argument is optional and is used to run several times the same experiments.

## Models available
This repository contains the code to train 
 * Simple neural networks
 * Deep ensembles and anchored ensembles (AE)
 * Sequential anchored ensembles (SAE)
 * Graphical anchored ensembles (an attempt to improve SAE not included in the paper)
 
A description of all those models can be found [here](models/).

## Cite our work
```
@article{delaunoy2021sae,
  title={SAE: Sequential Anchored Ensembles},
  author={Delaunoy, Arnaud and Louppe, Gilles},
  journal={arXiv preprint arXiv:2201.00649},
  year={2021}
}
```
