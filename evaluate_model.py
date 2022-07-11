import os
import argparse
import glob
import numpy as np
import yaml
from data import load_data
from models import load_model
from metrics import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default=None, help="name of the result files")
    parser.add_argument("--dataset_name", type=str, default=None, help="name of the dataset")

    args = vars(parser.parse_args())

    base_path = args["base_path"]

    result_dirs = glob.glob(base_path)

    for result_dir in result_dirs:
        #with open(os.path.join(result_dir, "config.yaml"), "r") as fp:
        #    config = yaml.load(fp, Loader=yaml.FullLoader)

        pred_probas = np.loadtxt(os.path.join(result_dir, "pred_probas.csv"))

        #dataset_name = config["dataset_name"]
        dataset_name = args["dataset_name"]
        datadir = os.path.join("data", dataset_name)
        true_probas = np.loadtxt(os.path.join(datadir, "probs.csv"))

        _, _, task = load_model(dataset_name)

        if task == "classification":
            agreement = compute_agreement(pred_probas, true_probas)
            total_variation = compute_total_variation(pred_probas, true_probas)
            with open(os.path.join(result_dir, "test_metrics.txt"), "w") as fp:
                fp.write("agreement: {}\n".format(agreement))
                fp.write("total_variation: {}\n".format(total_variation))

        elif task == "regression":
            w2 = w2_distance(pred_probas, np.transpose(true_probas))

            with open(os.path.join(result_dir, "test_metrics.txt"), "w") as fp:
                fp.write("w2: {}\n".format(w2))
