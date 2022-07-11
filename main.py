from data import load_data
from models import load_model, SimpleModel, EnsembleModel, SequentialEnsembleModel, GraphEnsembleModel, GraphGaussianEnsembleModel
from utils import evaluate_model
import argparse
import yaml
import os
import numpy as np
import tensorflow as tf
import wandb
import glob

def main(args):

    with open(args["config_file"], "r") as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    for key, value in args.items():
        config[key] = value

    args = config

    if "index" in args.keys():
        index = args["index"]
        if index is not None:
            args["save_dir"] = args["save_dir"] + "_{}".format(index)
    else:
        index = None

    dataset_name = args["dataset_name"]
    competition_mode = args["competition_mode"]

    if not competition_mode:
        wandb.init(project="bdl_competition_{}".format(dataset_name), entity="adelau", config=args)
        args = wandb.config
    print("args = {}".format(args))
    
    method = args["method"]
    train_batch_size = args["train_batch_size"]
    test_batch_size = args["test_batch_size"]
    nb_epochs = args["nb_epochs"]
    lr = args["lr"]
    min_lr = args["min_lr"]
    num_workers = args["num_workers"]
    save_dir = args["save_dir"]
    keep_best_weights = args["keep_best_weights"]
    optimizer = args["optimizer"]
    train_val_split = args["train_val_split"]
    normalize = args["normalize"]
    
    if "seed" in args.keys():
        seed = args["seed"]
        print("Using seed {}".format(seed))
    else:
        print("Running without seed")
        seed = None

    if "delete_files" not in args.keys():
        args["delete_files"] = False

    if "save_anchors" in args.keys():
        save_anchors = args["save_anchors"]
    else:
        save_anchors = False

    if "max_budget" in args.keys():
        max_budget = args["max_budget"]
    else:
        max_budget = None

    if "anchored" in args.keys():
        anchored = args["anchored"]
    else:
        anchored = True

    if "early_stopping" in args.keys():
        early_stopping = args["early_stopping"]
    else:
        early_stopping = False

    if "early_stopping_epochs" in args.keys():
        early_stopping_epochs = args["early_stopping_epochs"]
    else:
        early_stopping_epochs = 1

    if optimizer == "sgd":
        optimizer_args = {"optimizer_name": "sgd",
                          "momentum": args["momentum"],
                          "nesterov": args["nesterov"]}

    elif optimizer == "adam":
        optimizer_args = {"optimizer_name": "adam",
                          "b1": args["b1"],
                          "b2": args["b2"]}
    
    clip_gradient = args["clip_gradient"]
    increase_prior_variance = args["increase_prior_variance"]

    if clip_gradient:
        clipping_norm = args["clipping_norm"]
    else:
        # Not used
        clipping_norm = None

    imbalance = args["imbalance"]
    if imbalance:
        imbalance_factor = args["imbalance_factor"]
        imbalance_method = args["imbalance_method"]
        if imbalance_method == "focal":
            gamma = args["gamma"]
        else:
            #gamma = None
            gamma = args["gamma"]
    else:
        imbalance_method = None
        imbalance_factor = None
        gamma = None

    try: 
        os.mkdir("results")
    except FileExistsError:
        pass

    save_dir = os.path.join("results", save_dir)

    try: 
        os.mkdir(save_dir)
    except FileExistsError:
        pass
    
    train_set, val_set, test_set, probas = load_data(dataset_name, train_val_split=train_val_split, normalize=normalize, seed=seed)

    train_set_size = len(train_set)
    print("train_set_size = {}".format(train_set_size))

    train_loader = train_set.shuffle(1000, reshuffle_each_iteration=True)
    train_loader = train_loader.batch(train_batch_size)

    if val_set is not None:
        val_loader = val_set.batch(test_batch_size)
    else:
        val_loader = None
        
    test_loader = test_set.batch(test_batch_size)

    base_model, prior_variance, task = load_model(dataset_name)

    if method == "simple_model":
        model = SimpleModel(base_model, save_dir, task)

    if method == "ensemble":
        ensemble_size = args["ensemble_size"]
        model = EnsembleModel(base_model, save_dir, task, ensemble_size)

    if method == "sequential_ensemble":
        ensemble_size = args["ensemble_size"]
        num_chains = args["num_chains"]
        model = SequentialEnsembleModel(base_model, save_dir, task, ensemble_size, num_chains)

    if method == "graph_ensemble":
        ensemble_size = args["ensemble_size"]
        num_init_points = args["num_init_points"]
        distance = args["distance"]
        model = GraphEnsembleModel(base_model, save_dir, task, ensemble_size, num_init_points, distance)
    
    if method == "graph_gaussian_ensemble":
        ensemble_size = args["ensemble_size"]
        num_init_points = args["num_init_points"]
        posterior_approx = args["posterior_approx"]
        model = GraphGaussianEnsembleModel(base_model, save_dir, task, ensemble_size, num_init_points, posterior_approx)

    if method == "simple_model":
        model.train(train_loader, val_loader, nb_epochs, train_set_size, lr, min_lr, prior_variance*increase_prior_variance, keep_best_weights, optimizer_args, 
                    clip_gradient, clipping_norm, competition_mode, imbalance=imbalance, imbalance_factor=imbalance_factor, imbalance_method=imbalance_method, 
                    gamma=gamma, seed=seed, early_stopping=early_stopping, early_stopping_epochs=early_stopping_epochs, max_budget=max_budget)
    if method == "ensemble":
        model.train(train_loader, val_loader, nb_epochs, train_set_size, lr, min_lr, prior_variance*increase_prior_variance, keep_best_weights, optimizer_args, 
                    clip_gradient, clipping_norm, competition_mode, imbalance=imbalance, imbalance_factor=imbalance_factor, imbalance_method=imbalance_method, 
                    gamma=gamma, seed=seed, early_stopping=early_stopping, early_stopping_epochs=early_stopping_epochs, max_budget=max_budget, anchored=anchored)

    if method == "sequential_ensemble" or method == "graph_ensemble" or method == "graph_gaussian_ensemble":
        sequential_lr = args["sequential_lr"]
        sequential_min_lr = args["sequential_min_lr"]
        sequential_nb_epochs = args["sequential_nb_epochs"]

        sequential_optimizer = args["sequential_optimizer"]

        if sequential_optimizer == "sgd":
            sequential_optimizer_args = {"optimizer_name": "sgd",
                                         "momentum": args["sequential_momentum"],
                                         "nesterov": args["sequential_nesterov"]}

        elif sequential_optimizer == "adam":
            sequential_optimizer_args = {"optimizer_name": "adam",
                                         "b1": args["sequential_b1"],
                                         "b2": args["sequential_b2"]}

    if method == "sequential_ensemble":
        sampler = args["sampler"]

        if sampler == "iid":
            sampler_params = {"sampler": "iid"}

        if sampler == "gaussian_mh":
            sampler_params = {"sampler": "gaussian_mh",
                              "step_std": args["step_std"]}

        if sampler == "guided_walk":
            sampler_params = {"sampler": "guided_walk",
                              "step_std": args["step_std"]}

        if sampler == "hmc":
            sampler_params = {"sampler": "hmc",
                              "step_size": args["step_size"],
                              "num_steps": args["num_steps"]}


        model.train(train_loader, val_loader, nb_epochs, train_set_size, lr, min_lr, prior_variance*increase_prior_variance, keep_best_weights, optimizer_args, 
                    clip_gradient, clipping_norm, competition_mode, sampler_params, sequential_lr, sequential_min_lr, sequential_optimizer_args, 
                    sequential_nb_epochs, imbalance=imbalance, imbalance_factor=imbalance_factor, gamma=gamma, seed=seed, save_anchors=save_anchors,
                    early_stopping=early_stopping, early_stopping_epochs=early_stopping_epochs, max_budget=max_budget)

    if method == "graph_ensemble" or method == "graph_gaussian_ensemble":
                                         
        model.train(train_loader, val_loader, nb_epochs, train_set_size, lr, min_lr, prior_variance*increase_prior_variance, keep_best_weights, optimizer_args, 
                    clip_gradient, clipping_norm, competition_mode, sequential_lr, sequential_min_lr, sequential_optimizer_args, sequential_nb_epochs,
                    imbalance=imbalance, imbalance_factor=imbalance_factor, gamma=gamma, seed=seed, save_anchors=save_anchors, early_stopping=early_stopping, 
                    early_stopping_epochs=early_stopping_epochs, max_budget=max_budget)

    if task == "classification":
        if val_loader is not None and not competition_mode:
             _, _, _, accuracy, log_likelihood = evaluate_model(model, val_loader, None, test_batch_size, len(val_set), task)
        else:
            accuracy = 0.
            log_likelihood = 0.

        pred_probas, agreement, total_variation, _, _ = evaluate_model(model, test_loader, probas, test_batch_size, len(test_set), task)

        np.savetxt(os.path.join(save_dir, "pred_probas.csv"), pred_probas)
        if not competition_mode:
            np.savetxt(os.path.join(save_dir, "agreement.csv"), np.array([agreement]))
            np.savetxt(os.path.join(save_dir, "total_variation.csv"), np.array([total_variation]))
            np.savetxt(os.path.join(save_dir, "accuracy.csv"), np.array([accuracy]))
            np.savetxt(os.path.join(save_dir, "log_likelihood.csv"), np.array([log_likelihood]))

            with open(os.path.join(save_dir, "metrics.txt"), "w") as fp:
                fp.write("agreement: {}\n".format(agreement))
                fp.write("total_variation: {}\n".format(total_variation))
                fp.write("accuracy: {}\n".format(accuracy))
                fp.write("log_likelihood: {}".format(log_likelihood))

            print("Agreement = {}".format(agreement))
            print("Total variation = {}".format(total_variation))
            print("Accuracy = {}".format(accuracy))
            print("Log likelihood = {}".format(log_likelihood))

            wandb.log({"log_likelihood": log_likelihood.item()})

    if task == "regression":
        if val_loader is not None and not competition_mode:
            _, _, mse = evaluate_model(model, val_loader, None, test_batch_size, len(val_set), task)
        else:
            mse = 0.
        
        pred_probas, w2, _ = evaluate_model(model, test_loader, probas, test_batch_size, len(test_set), task)
        
        np.savetxt(os.path.join(save_dir, "pred_probas.csv"), pred_probas)

        if not competition_mode:
            np.savetxt(os.path.join(save_dir, "mse.csv"), np.array([mse]))
            np.savetxt(os.path.join(save_dir, "w2.csv"), np.array([w2]))

            with open(os.path.join(save_dir, "metrics.txt"), "w") as fp:
                fp.write("mse: {}\n".format(mse))
                fp.write("w2: {}\n".format(w2))

            print("mse = {}".format(mse))
            print("w2 = {}".format(w2))
            wandb.log({"mse": mse.item()})

    with open(os.path.join(save_dir, "config.yaml"), "w") as fp:
            yaml.dump(dict(args), fp)

    if args["delete_files"]:
        weight_files = glob.glob(os.path.join(save_dir, "weights*"))
        anchor_files = glob.glob(os.path.join(save_dir, "anchors*"))

        for file in weight_files + anchor_files:
            os.remove(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default=None, help="If a config file is given, all other arguments are ignored")
    parser.add_argument("--index", type=int, default=None, help="Index of the job to add to the save file")

    args = vars(parser.parse_args())

    main(args)