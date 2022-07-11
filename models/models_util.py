import os
from .jax_models import get_model

def load_model(dataset_name):
    if dataset_name == "cifar10":
        model = get_model("resnet20_frn_swish", data_info={"num_classes": 10})
        prior_variance = 1/5
        task="classification"

    if dataset_name == "imdb":
        model = get_model("cnn_lstm", data_info={"num_classes": 2})
        prior_variance = 1.
        task="classification"

    if dataset_name == "cifar_anon":
        model = get_model("cifar_alexnet", data_info={"num_classes": 10})
        prior_variance = 0.05
        task="classification"

    if dataset_name == "dermamnist_anon":
        model = get_model("medmnist_lenet", data_info={"num_classes": 7})
        prior_variance = 0.01
        task="classification"
    
    if dataset_name == "energy_anon":
        model = get_model("uci_mlp", {})
        prior_variance = 1 / 30
        task="regression"
    
    return model, prior_variance, task