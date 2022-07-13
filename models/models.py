import os
from .train import train_model
import math
from jax import numpy as jnp
import jax
import pickle
import numpy as np
from .iid_sampler import IIDSampler
from .guided_walk_gaussian_sampler import GuidedWalkGaussianSampler
from .gaussian_mh_sampler import GaussianMHSampler
from .hamiltonian_sampler import HamiltonianSampler
import glob

class SimpleModel():
    """
    Model composed of a single neural network
    """

    def __init__(self, base_model, save_dir, task):
        """Constructor

        Args:
            base_model (tuple): A tuple containing the init and apply functions returned by haiku.transform
            save_dir (str): The directory in which to save the weights
            task (str): The task performed, either "regression" or "classification"
        """

        self.save_dir = save_dir
        self.base_model_apply = jax.jit(base_model.apply)
        self.base_model_init = base_model.init
        self.task = task

    def train(self, train_loader, val_loader, nb_epochs, train_set_size, lr, min_lr, prior_variance, keep_best_weights, optimizer_args,
              competition_mode, seed=None, early_stopping=False, early_stopping_epochs=1, max_budget=None):
        """Train the model, weights will be saved in the save_dir directory.

        Args:
            train_loader (tf.data.Dataset): The train data loader
            val_loader (tf.data.Dataset): The validation data loader
            nb_epochs (int): Number of epochs to perform
            train_set_size (int): The size of the train set
            lr (float): The initial learning rate
            min_lr (float): The minimal learning rate reached by the scheduler
            prior_variance (float): The weight's prior variance
            keep_best_weights (bool): True to save the weights that achieved the best val loss, False to keep the last weights
            optimizer_args (dict): A dictionary containing the optimizers arguments (the optimizer name and corresponding parameters)
            competition_mode (bool): True to run in competition mode (no logging is performed)
            seed (int, optional): The seed used. Defaults to None.
            early_stopping (bool, optional): Set to True to perform early stopping. Defaults to False.
            early_stopping_epochs (int, optional): number of epochs without improvement required to stop. Defaults to 1.
            max_budget (int, optional): The maximal computational budget used (in epochs), pass None to not set any. Defaults to None.
        """

        # Apply seed if provided
        if seed is None:
            np.random.seed(None)
            seed = np.random.randint((1 << 63) - 1)

        # Initialize the weights
        key = jax.random.PRNGKey(seed)
        key, net_init_key = jax.random.split(key, 2)
        init_data, _ = next(iter(train_loader))
        init_data = jnp.asarray(init_data)
        params = self.base_model_init(net_init_key, init_data, True)

        # Set anchors to zeros to optimize for the MAP
        anchor = [jnp.full(p.size, 0.) for p in jax.tree_leaves(params)]

        # Train the neural network
        train_model(train_loader, val_loader, self.base_model_apply, params, nb_epochs, train_set_size, lr, min_lr, self.save_dir, anchor, prior_variance, keep_best_weights, 
                    optimizer_args, self.task, competition_mode, 0, early_stopping=early_stopping, 
                    early_stopping_epochs=early_stopping_epochs, max_budget=max_budget)

    def predict(self, data_loader):
        """Make predictions

        Args:
            data_loader (tf.data.Dataset): Data loader on which to make predictions

        Returns:
            np.array: 
                predicted probabilities if the task is "classification"
                samples from the predicted distribution if the task is "regression"
        """

        # Load weights
        params = pickle.load(open(os.path.join(self.save_dir, "weights_0.pkl"), "rb"))

        if self.task == "classification":
            # Compute predicted probabilities
            pred_logits = []

            for i, batch in enumerate(data_loader):
                x, _ = batch
                x = jnp.asarray(x)
                logits_tmp = self.base_model_apply(params, None, x, False)
                pred_logits.append(np.asarray(logits_tmp))

            pred_logits = jnp.concatenate(pred_logits, axis=0)
            pred_probas = jax.nn.softmax(pred_logits, axis=1)

            return pred_probas

        if self.task == "regression":
            pred_samples = []
            key = jax.random.PRNGKey(0)

            for i, batch in enumerate(data_loader):
                x, _ = batch
                x = jnp.asarray(x)

                # Compute predicted mean and std
                predictions = self.base_model_apply(params, None, x, False)
                predictions_mean, predictions_std = jnp.split(predictions, [1], axis=-1)
                predictions_std = jax.nn.softplus(predictions_std)

                # Sample from the predicted distribution
                key, subkey = jax.random.split(key)
                samples = jax.random.normal(subkey, (len(predictions_mean), 1000))
                samples = samples*predictions_std + predictions_mean

                pred_samples.append(samples)

            pred_samples = jnp.concatenate(pred_samples, axis=0)
            return pred_samples

class EnsembleModel():
    """
    Model composed of multiple (anchored) neural networks trained independently.
    Models anchored ensembles (AE) in the paper if anchored is set to True.
    """
    def __init__(self, base_model, save_dir, task, ensemble_size):
        """Constructor

        Args:
            base_model (tuple): A tuple containing the init and apply functions returned by haiku.transform
            save_dir (str): The directory in which to save the weights
            task (str): The task performed, either "regression" or "classification"
            ensemble_size (int): The size of the ensemble
        """

        self.save_dir = save_dir
        self.base_model_apply = jax.jit(base_model.apply)
        self.base_model_init = base_model.init
        self.ensemble_size = ensemble_size
        self.task = task

    def train(self, train_loader, val_loader, nb_epochs, train_set_size, lr, min_lr, prior_variance, keep_best_weights, optimizer_args,
              competition_mode, seed=None, early_stopping=False, early_stopping_epochs=1, max_budget=None, anchored=True):

        """Train the model, weights will be saved in the save_dir directory.

        Args:
            train_loader (tf.data.Dataset): The train data loader
            val_loader (tf.data.Dataset): The validation data loader
            nb_epochs (int): Number of epochs to perform
            train_set_size (int): The size of the train set
            lr (float): The initial learning rate
            min_lr (float): The minimal learning rate reached by the scheduler
            prior_variance (float): The weight's prior variance
            keep_best_weights (bool): True to save the weights that achieved the best val loss, False to keep the last weights
            optimizer_args (dict): A dictionary containing the optimizers arguments (the optimizer name and corresponding parameters)
            competition_mode (bool): True to run in competition mode (no logging is performed)
            seed (int, optional): The seed used. Defaults to None.
            early_stopping (bool, optional): Set to True to perform early stopping. Defaults to False.
            early_stopping_epochs (int, optional): number of epochs without improvement required to stop. Defaults to 1.
            max_budget (int, optional): The maximal computational budget used (in epochs), pass None to not set any. Defaults to None.
            anchored (bool, optional): Set to True to build an anchored ensemble and to False to build a deep ensemble. Defaults to True.
        """

        # Apply seed if provided
        if seed is None:
            np.random.seed(None)
            seed = np.random.randint((1 << 63) - 1)

        key = jax.random.PRNGKey(seed)

        total_epochs = 0
        epochs = []
        current_index = 0

        # Execute for each member of the ensemble
        for i in range(self.ensemble_size):
            print("training model {}/{}".format(i+1, self.ensemble_size))
            if max_budget is not None:
                    print("budget {}/{}".format(total_epochs, max_budget))

            # Initialize the weights
            key, net_init_key = jax.random.split(key, 2)
            init_data, _ = next(iter(train_loader))
            init_data = jnp.asarray(init_data)
            params = self.base_model_init(net_init_key, init_data, True)

            # Draw an anchor if the ensemble is anchored otherwise set it to zeros
            if anchored == True:
                anchor = []
                for p in jax.tree_leaves(params):
                    key, subkey = jax.random.split(key)
                    anchor.append(math.sqrt(prior_variance) * jax.random.normal(subkey, (p.size,)))
            else:
                anchor = [jnp.full(p.size, 0.) for p in jax.tree_leaves(params)]
            
            # If a max budget is specified, initialize the remaining budget
            if max_budget is not None:
                current_max_budget = max_budget - total_epochs
            else:
                current_max_budget = max_budget

            # Train the ensemble member
            params, epochs_made = train_model(train_loader, val_loader, self.base_model_apply, params, nb_epochs, train_set_size, lr, min_lr, self.save_dir, anchor, 
                                         prior_variance, keep_best_weights, optimizer_args, self.task, competition_mode, i, early_stopping=early_stopping, 
                                         early_stopping_epochs=early_stopping_epochs, max_budget=current_max_budget)

            if params is not None:
                epochs.append(epochs_made)
                current_index += 1
            
            total_epochs += epochs_made

            # If max budget is specified and reached, stop the training 
            if max_budget is not None and total_epochs >= max_budget:
                break
        
        # Save the number of members that have effectively been trained
        self.effective_ensemble_size = current_index

        # Save the number of epochs performed for each ensemble member.
        np.save(os.path.join(self.save_dir, "epochs.npy"), np.array(epochs))

    def predict(self, data_loader):
        """Make predictions

        Args:
            data_loader (tf.data.Dataset): Data loader on which to make predictions

        Returns:
            np.array: 
                predicted probabilities if the task is "classification"
                samples from the predicted distribution if the task is "regression"
        """

        if self.task == "classification":
            pred_probas = None

            # Make predictions with each ensemble member
            for i in range(self.effective_ensemble_size):
                print("predicting with ensemble {}/{}".format(i+1, self.effective_ensemble_size))

                params = pickle.load(open(os.path.join(self.save_dir, "weights_{}.pkl".format(i)), "rb"))
                
                pred_logits = []

                # Make the predictions for this member
                for i, batch in enumerate(data_loader):
                    x, _ = batch
                    x = jnp.asarray(x)
                    logits_tmp = self.base_model_apply(params, None, x, False)
                    pred_logits.append(np.asarray(logits_tmp))

                pred_logits = jnp.concatenate(pred_logits, axis=0)

                tmp_probas = jax.nn.softmax(pred_logits, axis=1)

                # Add the prediction of this member to the others
                if pred_probas is None:
                    pred_probas = tmp_probas
                else:
                    pred_probas += tmp_probas

            # Average the predictions
            return pred_probas/self.effective_ensemble_size

        if self.task == "regression":
            pred_samples = []

            # Compute the number of samples to drawn with each member to reach a total of 1000 samples
            nb_samples_per_estimator = 1000//self.effective_ensemble_size
            rest = 1000 - nb_samples_per_estimator * self.effective_ensemble_size

            # Draw samples with each member
            for i in range(self.effective_ensemble_size):
                print("predicting with ensemble {}/{}".format(i+1, self.effective_ensemble_size))

                params = pickle.load(open(os.path.join(self.save_dir, "weights_{}.pkl".format(i)), "rb"))
                key = jax.random.PRNGKey(0)

                tmp_samples = []

                for j, batch in enumerate(data_loader):
                    x, _ = batch
                    x = jnp.asarray(x)

                    # Compute predicted mean and std
                    predictions = self.base_model_apply(params, None, x, False)
                    predictions_mean, predictions_std = jnp.split(predictions, [1], axis=-1)
                    predictions_std = jax.nn.softplus(predictions_std)

                    # Sample from the predicted distribution
                    key, subkey = jax.random.split(key)
                    if i < rest:
                        samples = jax.random.normal(subkey, (len(predictions_mean), nb_samples_per_estimator+1))
                    else:
                        samples = jax.random.normal(subkey, (len(predictions_mean), nb_samples_per_estimator))

                    samples = samples*predictions_std + predictions_mean

                    tmp_samples.append(samples)

                # Concatenate the samples of each batch
                tmp_samples = jnp.concatenate(tmp_samples, axis=0)
                pred_samples.append(tmp_samples)

            # Concatenate the samples of each member
            pred_samples = jnp.concatenate(pred_samples, axis=1)

            return pred_samples
        
class SequentialEnsembleModel():
    """
    Model composed of multiple anchored neural networks trained sequentially.
    Models sequential anchored ensembles (SAE) in the paper.
    """
    def __init__(self, base_model, save_dir, task, ensemble_size, num_chains):
        """Constructor

        Args:
            base_model (tuple): A tuple containing the init and apply functions returned by haiku.transform
            save_dir (str): The directory in which to save the weights
            task (str): The task performed, either "regression" or "classification"
            ensemble_size (int): The size of the ensemble
            num_chains (int): The number of chains to run
        """

        self.save_dir = save_dir
        self.base_model_apply = jax.jit(base_model.apply)
        self.base_model_init = base_model.init
        self.ensemble_size = ensemble_size
        self.num_chains = num_chains
        self.num_estimators = self.ensemble_size * self.num_chains
        self.task = task

    def train(self, train_loader, val_loader, nb_epochs, train_set_size, lr, min_lr, prior_variance, keep_best_weights, optimizer_args, 
              competition_mode, sampler_params, sequential_lr, sequential_min_lr, sequential_optimizer_args, sequential_nb_epochs, seed=None, 
              save_anchors=False, early_stopping=False, early_stopping_epochs=1, max_budget=None):
        """Train the model, weights will be saved in the save_dir directory.

        Args:
            train_loader (tf.data.Dataset): The train data loader
            val_loader (tf.data.Dataset): The validation data loader
            nb_epochs (int): Number of epochs to perform for the first training
            train_set_size (int): The size of the train set
            lr (float): The initial learning rate for the first training
            min_lr (float): The minimal learning rate reached by the scheduler for the first training
            prior_variance (float): The weight's prior variance
            keep_best_weights (bool): True to save the weights that achieved the best val loss, False to keep the last weights
            optimizer_args (dict): A dictionary containing the optimizers arguments for the first training (the optimizer name and corresponding parameters)
            competition_mode (bool): True to run in competition mode (no logging is performed)
            sampler_params (dict): A dictionary containing the anchor sampler arguments (the sampler name and corresponding parameters)
            sequential_lr (float): The initial learning rate for the sequential trainings
            sequential_min_lr (float): The minimal learning rate reached by the scheduler for the sequential trainings
            sequential_optimizer_args (dict): A dictionary containing the optimizers arguments for the sequential trainings (the optimizer name and corresponding parameters)
            sequential_nb_epochs (int): Number of epochs to perform for the sequential trainings
            seed (int, optional): The seed used. Defaults to None.
            save_anchors (bool, optional): Set to True to save the anchors in the save_dir directory. Defaults to False.
            early_stopping (bool, optional): Set to True to perform early stopping. Defaults to False.
            early_stopping_epochs (int, optional): number of epochs without improvement required to stop. Defaults to 1.
            max_budget (int, optional): The maximal computational budget used (in epochs), pass None to not set any. Defaults to None.
        """
        
        # Apply seed if provided
        if seed is None:
            np.random.seed(None)
            seed = np.random.randint((1 << 63) - 1)

        key = jax.random.PRNGKey(seed)

        # If a maximal budget is specified, split it between the different chains
        if max_budget is None:
            max_chain_budget = None
        else:
            max_chain_budget = max_budget//self.num_chains

        current_index = 0

        # Execute fo each chain
        for chain in range(self.num_chains):
            epochs = []
            print("running chain {}/{}".format(chain+1, self.num_chains))

            total_epochs = 0

            # If a max budget is specified, initialize the remaining budget
            if max_budget is not None:
                current_max_budget = max_chain_budget - total_epochs
            else:
                current_max_budget = max_chain_budget

            # Initialize the weights
            key, net_init_key = jax.random.split(key, 2)
            init_data, _ = next(iter(train_loader))
            init_data = jnp.asarray(init_data)
            params = self.base_model_init(net_init_key, init_data, True)

            # Initialize the anchor sampler
            if sampler_params["sampler"] == "iid":
                anchor_samplers = [IIDSampler(np.full((p.size,), 0.), np.full((p.size,), math.sqrt(prior_variance))) for p in jax.tree_leaves(params)]
            if sampler_params["sampler"] == "guided_walk":
                anchor_samplers = [GuidedWalkGaussianSampler(np.full((p.size,), 0.), np.full((p.size,), math.sqrt(prior_variance)), sampler_params["step_std"]) for p in jax.tree_leaves(params)]
            if sampler_params["sampler"] == "gaussian_mh":
                anchor_samplers = [GaussianMHSampler(np.full((p.size,), 0.), np.full((p.size,), math.sqrt(prior_variance)), sampler_params["step_std"]) for p in jax.tree_leaves(params)]
            if sampler_params["sampler"] == "hmc":
                anchor_samplers = [HamiltonianSampler(np.full((p.size,), 0.), np.full((p.size,), math.sqrt(prior_variance)), sampler_params["step_size"], sampler_params["num_steps"]) for p in jax.tree_leaves(params)]

            # Draw anchors
            anchor = [jnp.asarray(x.sample()) for x in anchor_samplers]
            if save_anchors:
                with open(os.path.join(self.save_dir, "anchors_{}.pkl".format(self.ensemble_size*chain)), "wb") as fp:
                    pickle.dump(anchor, fp)

            # Train the first model
            print("training model {}/{}".format(1, self.ensemble_size))
            params, epochs_made = train_model(train_loader, val_loader, self.base_model_apply, params, nb_epochs, train_set_size, lr, min_lr, self.save_dir, anchor, prior_variance, 
                                              keep_best_weights, optimizer_args, self.task, competition_mode, current_index, max_gamma=gamma, early_stopping=early_stopping, 
                                              early_stopping_epochs=early_stopping_epochs, max_budget=current_max_budget)
            
            total_epochs += epochs_made
            if params is not None:
                current_index += 1
                epochs.append(epochs_made)

            # If a maximal budget is specified and reached, stop the training
            if max_chain_budget is not None and total_epochs >= max_chain_budget:
                self.effective_ensemble_size = current_index
                np.save(os.path.join(self.save_dir, "epochs_chain_{}.npy".format(chain)), np.array(epochs))
                continue
            
            # Execute for each additional member to train
            for i in range(1, self.ensemble_size):
                print("training model {}/{}".format(i+1, self.ensemble_size))
                if max_chain_budget is not None:
                    print("budget {}/{}".format(total_epochs, max_chain_budget))

                # If a max budget is specified, initialize the remaining budget
                if max_budget is not None:
                    current_max_budget = max_chain_budget - total_epochs
                else:
                    current_max_budget = max_chain_budget

                # Draw next anchors
                anchor = [jnp.asarray(x.sample()) for x in anchor_samplers]
                if save_anchors:
                    with open(os.path.join(self.save_dir, "anchors_{}.pkl".format(self.ensemble_size*chain+i)), "wb") as fp:
                        pickle.dump(anchor, fp)
                
                # Train the model by starting from the weight obtained at the previous iteration
                params, epochs_made = train_model(train_loader, val_loader, self.base_model_apply, params, sequential_nb_epochs, train_set_size, sequential_lr, sequential_min_lr, 
                                                  self.save_dir, anchor, prior_variance, keep_best_weights, sequential_optimizer_args, self.task, competition_mode, 
                                                  current_index, early_stopping=early_stopping, early_stopping_epochs=early_stopping_epochs, 
                                                  max_budget=current_max_budget)

                total_epochs += epochs_made
                if params is not None:
                    current_index += 1
                    epochs.append(epochs_made)

                # If a maximal budget is specified and reached, stop the training
                if max_chain_budget is not None and total_epochs >= max_chain_budget:
                    break
            
            # Save the number of epochs performed for each ensemble member.
            np.save(os.path.join(self.save_dir, "epochs_chain_{}.npy".format(chain)), np.array(epochs))
        
        # Save the number of members that have effectively been trained
        self.effective_num_estimators = current_index

    def predict(self, data_loader):
        """Make predictions

        Args:
            data_loader (tf.data.Dataset): Data loader on which to make predictions

        Returns:
            np.array: 
                predicted probabilities if the task is "classification"
                samples from the predicted distribution if the task is "regression"
        """
        if self.task == "classification":
            pred_probas = None

            # Make predictions with each ensemble member
            for i in range(self.effective_num_estimators):
                print("predicting with ensemble {}/{}".format(i+1, self.effective_num_estimators))

                params = pickle.load(open(os.path.join(self.save_dir, "weights_{}.pkl".format(i)), "rb"))
                
                pred_logits = []

                # Make the predictions for this member
                for i, batch in enumerate(data_loader):
                    x, _ = batch
                    x = jnp.asarray(x)
                    logits_tmp = self.base_model_apply(params, None, x, False)
                    pred_logits.append(np.asarray(logits_tmp))

                pred_logits = jnp.concatenate(pred_logits, axis=0)

                tmp_probas = jax.nn.softmax(pred_logits, axis=1)

                # Add the prediction of this member to the others
                if pred_probas is None:
                    pred_probas = tmp_probas
                else:
                    pred_probas += tmp_probas

            # Average the predictions
            return pred_probas/self.effective_num_estimators

        if self.task == "regression":
            pred_samples = []

            # Compute the number of samples to drawn with each member to reach a total of 1000 samples
            nb_samples_per_estimator = 1000//self.effective_num_estimators
            rest = 1000 - nb_samples_per_estimator * self.effective_num_estimators

            # Draw samples with each member
            for i in range(self.effective_num_estimators):
                print("predicting with ensemble {}/{}".format(i+1, self.effective_num_estimators))

                params = pickle.load(open(os.path.join(self.save_dir, "weights_{}.pkl".format(i)), "rb"))
                key = jax.random.PRNGKey(0)

                tmp_samples = []

                for j, batch in enumerate(data_loader):
                    x, _ = batch
                    x = jnp.asarray(x)

                    # Compute predicted mean and std
                    predictions = self.base_model_apply(params, None, x, False)
                    predictions_mean, predictions_std = jnp.split(predictions, [1], axis=-1)
                    predictions_std = jax.nn.softplus(predictions_std)

                    # Sample from the predicted distribution
                    key, subkey = jax.random.split(key)
                    if i < rest:
                        samples = jax.random.normal(subkey, (len(predictions_mean), nb_samples_per_estimator+1))
                    else:
                        samples = jax.random.normal(subkey, (len(predictions_mean), nb_samples_per_estimator))

                    samples = samples*predictions_std + predictions_mean

                    tmp_samples.append(samples)

                # Concatenate the samples of each batch
                tmp_samples = jnp.concatenate(tmp_samples, axis=0)
                pred_samples.append(tmp_samples)

            # Concatenate the samples of each member
            pred_samples = jnp.concatenate(pred_samples, axis=1)

            return pred_samples

def l1_distance(params1, params2):
    """Compute the L1 distance between two params

    Args:
        params1 (list of jnp.array): The first parameters
        params2 (list of jnp.array): The second parameters

    Returns:
        float: The L1 distance
    """
    return jnp.sum(jnp.abs(jnp.concatenate(params1) - jnp.concatenate(params2)))

def l2_distance(params1, params2):
    """Compute the L2 distance between two params

    Args:
        params1 (list of jnp.array): The first parameters
        params2 (list of jnp.array): The second parameters

    Returns:
        float: The L2 distance
    """
    return jnp.sum((jnp.concatenate(params1) - jnp.concatenate(params2))**2)

class GraphEnsembleModel():
    """
    Not included in the paper because it does not perform better than SequentialEnsembleModel.
    Construct an anchored ensemble sequentially by first sampling anchors independently and then training the members sequentially in the order
    that minimizes the distance between consecutive anchors.
    """
    def __init__(self, base_model, save_dir, task, ensemble_size, num_init_points, distance):
        """Constructor

        Args:
            base_model (tuple): A tuple containing the init and apply functions returned by haiku.transform
            save_dir (str): The directory in which to save the weights
            task (str): The task performed, either "regression" or "classification"
            ensemble_size (int): The size of the ensemble
            num_init_points (int): The number of members trained from scratch.
            distance (str): The distance function that compute the distance between two anchors, either "l1" or "l2".
        """

        self.save_dir = save_dir
        self.base_model_apply = jax.jit(base_model.apply)
        self.base_model_init = base_model.init
        self.ensemble_size = ensemble_size
        self.num_init_points = num_init_points
        self.num_estimators = self.ensemble_size
        self.task = task
        self.distance = distance

        if distance == "l1":
            self.distance_fct = l1_distance

        if distance == "l2":
            self.distance_fct = l2_distance

    def train(self, train_loader, val_loader, nb_epochs, train_set_size, lr, min_lr, prior_variance, keep_best_weights, optimizer_args, 
              competition_mode, sequential_lr, sequential_min_lr, sequential_optimizer_args, sequential_nb_epochs, seed=None, save_anchors=False, 
              early_stopping=False, early_stopping_epochs=1, max_budget=None):
        """Train the model, weights will be saved in the save_dir directory.

        Args:
            train_loader (tf.data.Dataset): The train data loader
            val_loader (tf.data.Dataset): The validation data loader
            nb_epochs (int): Number of epochs to perform for the first training
            train_set_size (int): The size of the train set
            lr (float): The initial learning rate for the first training
            min_lr (float): The minimal learning rate reached by the scheduler for the first training
            prior_variance (float): The weight's prior variance
            keep_best_weights (bool): True to save the weights that achieved the best val loss, False to keep the last weights
            optimizer_args (dict): A dictionary containing the optimizers arguments for the first training (the optimizer name and corresponding parameters)
            competition_mode (bool): True to run in competition mode (no logging is performed)
            sequential_lr (float): The initial learning rate for the sequential trainings
            sequential_min_lr (float): The minimal learning rate reached by the scheduler for the sequential trainings
            sequential_optimizer_args (dict): A dictionary containing the optimizers arguments for the sequential trainings (the optimizer name and corresponding parameters)
            sequential_nb_epochs (int): Number of epochs to perform for the sequential trainings
            seed (int, optional): The seed used. Defaults to None.
            save_anchors (bool, optional): Set to True to save the anchors in the save_dir directory. Defaults to False.
            early_stopping (bool, optional): Set to True to perform early stopping. Defaults to False.
            early_stopping_epochs (int, optional): number of epochs without improvement required to stop. Defaults to 1.
            max_budget (int, optional): The maximal computational budget used (in epochs), pass None to not set any. Defaults to None.
        """
        
        # Apply seed if provided
        if seed is None:
            np.random.seed(None)
            seed = np.random.randint((1 << 31) - 1)

        np.random.seed(seed)

        key = jax.random.PRNGKey(seed)

        # Draw anchors
        key, net_init_key = jax.random.split(key, 2)
        init_data, _ = next(iter(train_loader))
        init_data = jnp.asarray(init_data)
        params = self.base_model_init(net_init_key, init_data, True)

        print("Drawing anchors")
        for i in range(self.num_estimators):
            
            anchor = []
            for p in jax.tree_leaves(params):
                key, subkey = jax.random.split(key)
                anchor.append(math.sqrt(prior_variance) * jax.random.normal(subkey, (p.size,)))

            with open(os.path.join(self.save_dir, "anchors_{}.pkl".format(i)), "wb") as fp:
                pickle.dump(anchor, fp)
        
        # Compute distance matrix
        print("Computing distance matrix")
        distance_matrix = np.empty((self.num_estimators, self.num_estimators))
        
        for anchor_index_1 in range(self.num_estimators):
            anchor_1 = pickle.load(open(os.path.join(self.save_dir, "anchors_{}.pkl".format(anchor_index_1)), "rb"))
            for anchor_index_2 in range(self.num_estimators):
                anchor_2 = pickle.load(open(os.path.join(self.save_dir, "anchors_{}.pkl".format(anchor_index_2)), "rb"))

                distance = self.distance_fct(anchor_1, anchor_2)
                distance_matrix[anchor_index_1][anchor_index_2] = distance
                distance_matrix[anchor_index_2][anchor_index_1] = distance

        epochs = []
        total_epochs = 0
        self.training_performed = np.full((self.num_estimators,), False)

        # Train initial neural networks
        print("Training initial networks")
        initial_anchors = np.random.choice(np.arange(self.num_estimators), size=self.num_init_points, replace=False)

        for i, anchor_index in enumerate(initial_anchors):
            # If a max budget is specified, initialize the remaining budget
            if max_budget is None:
                current_max_budget = None
            else:
                current_max_budget = max_budget - total_epochs

            print("training initial model {}/{}".format(i+1, self.num_init_points))
            if max_budget is not None:
                    print("budget {}/{}".format(total_epochs, max_budget))

            # Load the anchor
            anchor = pickle.load(open(os.path.join(self.save_dir, "anchors_{}.pkl".format(anchor_index)), "rb"))
            
            # Train the model
            params, epochs_made = train_model(train_loader, val_loader, self.base_model_apply, params, nb_epochs, train_set_size, lr, min_lr, self.save_dir, anchor, prior_variance, keep_best_weights, 
                                              optimizer_args, self.task, competition_mode, anchor_index, early_stopping=early_stopping, 
                                              early_stopping_epochs=early_stopping_epochs, max_budget=current_max_budget)
            
            total_epochs += epochs_made
            if params is not None:
                self.training_performed[anchor_index] = True
                epochs.append(epochs_made)
            
            # If a maximal budget is specified and reached, stop the training
            if max_budget is not None and total_epochs >= max_budget:
                    break

        # Compute training order
        print("Computing training order")
        
        # Vector retaining which models have already been trained
        trained = np.full((self.num_estimators,), False)
        trained[initial_anchors] = True

        # List containing the order of training
        compute_list = np.empty((self.num_estimators - self.num_init_points,), dtype=np.int32)
        # List containing the corresponding starting points
        starting_points = np.empty((self.num_estimators - self.num_init_points,), dtype=np.int32)

        for i in range(self.num_estimators - self.num_init_points):
            trained_indices = np.where(trained)[0]
            untrained_indices = np.where(np.logical_not(trained))[0]

            # Retrieve distance between all trained and untrained models
            shrinked_distance_matrix = distance_matrix[untrained_indices, :][: ,trained_indices]

            # Get the trained/untrained model pair with minimal anchor distance
            min_distance_matrix = np.min(shrinked_distance_matrix, axis=1)
            min_index = untrained_indices[np.argmin(min_distance_matrix)]

            # Next model to be trained in the untrained model of this pair
            compute_list[i] = min_index
            # Corresponding starting point is the trained model of this pair
            starting_points[i] = trained_indices[np.argmin(distance_matrix[min_index, :][trained_indices])]

            # Mark the corresponding model as trained
            trained[min_index] = True

        # Train the other neural networks
        print("Training other networks")
        for i in range(self.num_estimators - self.num_init_points):
            # If a maximal budget is specified and reached, stop the training
            if max_budget is not None and total_epochs >= max_budget:
                    break
            print("training sequential model {}/{}".format(i+1, self.num_estimators - self.num_init_points))
            if max_budget is not None:
                    print("budget {}/{}".format(total_epochs, max_budget))
            
            # Get the anchor of the next model to compute and the corresponding starting point
            starting_point_index = starting_points[i]
            anchor_index = compute_list[i]

            anchor = pickle.load(open(os.path.join(self.save_dir, "anchors_{}.pkl".format(anchor_index)), "rb"))
            starting_weights = pickle.load(open(os.path.join(self.save_dir, "weights_{}.pkl".format(starting_point_index)), "rb"))

            # If a max budget is specified, initialize the remaining budget
            if max_budget is None:
                current_max_budget = None
            else:
                current_max_budget = max_budget - total_epochs

            # Train the model
            params, epochs_made = train_model(train_loader, val_loader, self.base_model_apply, starting_weights, sequential_nb_epochs, train_set_size, sequential_lr, sequential_min_lr, 
                                              self.save_dir, anchor, prior_variance, keep_best_weights, sequential_optimizer_args, self.task, competition_mode, 
                                              anchor_index, early_stopping=early_stopping, early_stopping_epochs=early_stopping_epochs, max_budget=current_max_budget)

            total_epochs += epochs_made
            if params is not None:
                self.training_performed[anchor_index] = True
                epochs.append(epochs_made)

        if not save_anchors:
            #delete anchors
            anchor_files = glob.glob(os.path.join(self.save_dir, "anchors*"))
            for file in anchor_files:
                os.remove(file)
        
        # Retain the models that have been trained
        self.trained = trained

        # Save the number of epochs performed for each ensemble member.
        np.save(os.path.join(self.save_dir, "epochs.npy"), np.array(epochs))


    def predict(self, data_loader):
        """Make predictions

        Args:
            data_loader (tf.data.Dataset): Data loader on which to make predictions

        Returns:
            np.array: 
                predicted probabilities if the task is "classification"
                samples from the predicted distribution if the task is "regression"
        """
        trained_indices = np.where(self.training_performed)[0]
        print("trained indices = {}".format(trained_indices))
        effective_num_estimators = len(trained_indices)

        if self.task == "classification":
            pred_probas = None

            # Make predictions with each ensemble member
            for i in range(self.num_estimators):

                # Only predict if the model has actually been trained
                if self.training_performed[i]:
                    print("predicting with ensemble {}/{}".format(i+1, self.num_estimators))

                    params = pickle.load(open(os.path.join(self.save_dir, "weights_{}.pkl".format(i)), "rb"))
                    
                    pred_logits = []

                    # Make the predictions for this member
                    for i, batch in enumerate(data_loader):
                        x, _ = batch
                        x = jnp.asarray(x)
                        logits_tmp = self.base_model_apply(params, None, x, False)
                        pred_logits.append(np.asarray(logits_tmp))

                    pred_logits = jnp.concatenate(pred_logits, axis=0)

                    tmp_probas = jax.nn.softmax(pred_logits, axis=1)

                    # Add the prediction of this member to the others
                    if pred_probas is None:
                        pred_probas = tmp_probas
                    else:
                        pred_probas += tmp_probas

            # Average the predictions
            return pred_probas/effective_num_estimators

        if self.task == "regression":
            pred_samples = []
            
            # Compute the number of samples to drawn with each member to reach a total of 1000 samples
            nb_samples_per_estimator = 1000//effective_num_estimators
            rest = 1000 - nb_samples_per_estimator * effective_num_estimators

            # Draw samples with each member
            for i in range(effective_num_estimators):
                print("predicting with ensemble {}/{}".format(i+1, effective_num_estimators))

                params = pickle.load(open(os.path.join(self.save_dir, "weights_{}.pkl".format(trained_indices[i])), "rb"))
                key = jax.random.PRNGKey(0)

                tmp_samples = []

                for j, batch in enumerate(data_loader):
                    x, _ = batch
                    x = jnp.asarray(x)

                    # Compute predicted mean and std
                    predictions = self.base_model_apply(params, None, x, False)
                    predictions_mean, predictions_std = jnp.split(predictions, [1], axis=-1)
                    predictions_std = jax.nn.softplus(predictions_std)

                    # Sample from the predicted distribution
                    key, subkey = jax.random.split(key)
                    if i < rest:
                        samples = jax.random.normal(subkey, (len(predictions_mean), nb_samples_per_estimator+1))
                    else:
                        samples = jax.random.normal(subkey, (len(predictions_mean), nb_samples_per_estimator))

                    samples = samples*predictions_std + predictions_mean

                    tmp_samples.append(samples)

                # Concatenate the samples of each batch
                tmp_samples = jnp.concatenate(tmp_samples, axis=0)
                pred_samples.append(tmp_samples)

            # Concatenate the samples of each member
            pred_samples = jnp.concatenate(pred_samples, axis=1)

            return pred_samples
