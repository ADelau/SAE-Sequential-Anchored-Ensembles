import os
from .jax_models import get_model


def load_model(dataset_name):
    if dataset_name == "cifar10" or dataset_name == "cifar10test":

        model = get_model("resnet20_frn_swish", data_info={"num_classes": 10})
        prior_variance = 1/5
        task="classification"

    if dataset_name == "imdb" or dataset_name == "imdbtest":
        model = get_model("cnn_lstm", data_info={"num_classes": 2})
        prior_variance = 1.
        task="classification"

    if dataset_name == "toy":
        
        model = get_model("mlp_classification_toy", data_info={"num_classes": 3})
        prior_variance = 10.
        task="classification"

    if dataset_name == "retinopathy" or dataset_name == "retinopathy_test":

        model = get_model("retinopathy_cnn", data_info={"num_classes": 2})
        prior_variance = 0.05
        task="classification"

    if dataset_name == "cifar_anon" or dataset_name == "cifar_anon_test":
        model = get_model("cifar_alexnet", data_info={"num_classes": 10})
        prior_variance = 0.05
        task="classification"

    if dataset_name == "dermamnist_anon" or dataset_name == "dermamnist_anon_test":
        model = get_model("medmnist_lenet", data_info={"num_classes": 7})
        prior_variance = 0.01
        task="classification"

    if dataset_name == "energy_anon_1_1":
        model = get_model("uci_mlp", {})
        prior_variance = 1.
        task="regression"
    
    if dataset_name == "energy_anon_1_2":
        model = get_model("uci_mlp", {})
        prior_variance = 1 / 2
        task="regression"
    
    if dataset_name == "energy_anon_1_4":
        model = get_model("uci_mlp", {})
        prior_variance = 1 / 4
        task="regression"
    
    if dataset_name == "energy_anon_1_8":
        model = get_model("uci_mlp", {})
        prior_variance = 1 / 8
        task="regression"

    if dataset_name == "energy_anon_1_15":
        model = get_model("uci_mlp", {})
        prior_variance = 1 / 15
        task="regression"

    if dataset_name == "energy_anon_1_30":
        model = get_model("uci_mlp", {})
        prior_variance = 1 / 30
        task="regression"

    if dataset_name == "energy_anon_1_60":
        model = get_model("uci_mlp", {})
        prior_variance = 1 / 60
        task="regression"
    
    if dataset_name == "energy_anon_1_120":
        model = get_model("uci_mlp", {})
        prior_variance = 1 / 120
        task="regression"

    if dataset_name == "energy_anon_1_240":
        model = get_model("uci_mlp", {})
        prior_variance = 1 / 240
        task="regression"

    if dataset_name == "energy_anon_1_480":
        model = get_model("uci_mlp", {})
        prior_variance = 1 / 480
        task="regression"

    return model, prior_variance, task