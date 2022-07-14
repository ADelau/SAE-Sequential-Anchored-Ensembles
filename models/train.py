import os
import math
import copy
from jax import numpy as jnp
import jax
import numpy as np
import optax
import pickle
import wandb

"""File inspired from https://github.com/izmailovpavel/neurips_bdl_starter_kit"""

# Add gradient clipping
def train_model(train_loader, val_loader, model_apply, params, nb_epochs, train_set_size, init_lr, min_lr, save_dir, anchor, prior_variance, keep_best_weights, 
                optimizer_args, task, competition_mode, model_index, early_stopping=False, early_stopping_epochs=1, max_budget=None):
    """Train the neural network, weights will be saved in the save_dir directory.

        Args:
            train_loader (tf.data.Dataset): The train data loader
            val_loader (tf.data.Dataset): The validation data loader
            model_apply (function): The function modelling the neural network
            nb_epochs (int): Number of epochs to perform
            train_set_size (int): The size of the train set
            init_lr (float): The initial learning rate
            min_lr (float): The minimal learning rate reached by the scheduler
            save_dir (str): The directory in which to save the weights
            anchor (list of jnp.array): anchors used to regularize
            prior_variance (float): The weight's prior variance
            keep_best_weights (bool): True to save the weights that achieved the best val loss, False to keep the last weights
            optimizer_args (dict): A dictionary containing the optimizers arguments (the optimizer name and corresponding parameters)
            task (str): The task to perform, either "regression" or "classification"
            competition_mode (bool): True to run in competition mode (no logging is performed)
            model_index (int): The index of the model (e.g. in an ensemble)
            early_stopping (bool, optional): Set to True to perform early stopping. Defaults to False.
            early_stopping_epochs (int, optional): number of epochs without improvement required to stop. Defaults to 1.
            max_budget (int, optional): The maximal computational budget used (in epochs), pass None to not set any. Defaults to None.
        """

    # Create the save directory if not exist
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass
        

    def log_likelihood_fct_classif(params, batch, is_training=True):
        """Evaluate the log likelihood of a batch given some neural network parameters for a classification task.

        Args:
            params (jnp.array): The parameters of the neural network
            batch (tuple of two jnp.array): (x,y) representing the data in the batch
            is_training (bool, optional): Whether this is run in training mode. Defaults to True.

        Returns:
            float: The log likelihood
        """

        # Extract data
        x, y = batch

        # Evaluate the neural network
        logits = model_apply(params, None, x, is_training)
        num_classes = logits.shape[-1]
        labels = jax.nn.one_hot(y, num_classes)

        # Evaluate the likelihood
        softmax_xent = jnp.mean(jnp.sum(labels * jax.nn.log_softmax(logits), axis=1)) * train_set_size
        
        return softmax_xent

    def log_likelihood_fct_regression(params, batch, is_training=True):
        """Evaluate the log likelihood of a batch given some neural network parameters for a regression task.

        Args:
            params (jnp.array): The parameters of the neural network
            batch (tuple of two jnp.array): (x,y) representing the data in the batch
            is_training (bool, optional): Whether this is run in training mode. Defaults to True.

        Returns:
            float: The log likelihood
        """

        # Extract data
        x, y = batch

        # Evaluate the neural network
        predictions = model_apply(params, None, x, is_training)
        predictions_mean, predictions_std = jnp.split(predictions, [1], axis=-1)
        predictions_std = jax.nn.softplus(predictions_std)

        # Evaluate the likelihood
        se = (predictions_mean - y)**2
        log_likelihood = (-0.5 * se / predictions_std**2 -
                          0.5 * jnp.log(predictions_std**2 * 2 * math.pi))
        log_likelihood = jnp.mean(log_likelihood) * train_set_size

        return log_likelihood


    def log_prior_fct(params):
        """Evaluate the log prior of some neural network parameters

        Args:
            params (jnp.array): The parameters of the neural network

        Returns:
            float: The log prior
        """
        n_params = sum([p.size for p in jax.tree_leaves(params)])
        exp_term = sum([(-(jnp.ravel(p)-a)**2 / (2 * prior_variance)).sum() for p, a in zip(jax.tree_leaves(params), anchor)])
        norm_constant = -0.5 * n_params * jnp.log((2 * math.pi * prior_variance))
        return exp_term + norm_constant

    
    if task == "classification":
        @jax.jit
        def log_posterior_fct(params, batch, is_training=True):
            """Evaluate the log posterior of a batch given some neural network parameters for a classification task.

            Args:
                params (jnp.array): The parameters of the neural network
                batch (tuple of two jnp.array): (x,y) representing the data in the batch
                is_training (bool, optional): Whether this is run in training mode. Defaults to True.

            Returns:
                float: The log posterior
            """
            log_lik = log_likelihood_fct_classif(params, batch, is_training=True)
            log_prior = log_prior_fct(params)
            return log_lik + log_prior

    if task == "regression":
        @jax.jit
        def log_posterior_fct(params, batch, is_training=True):
            """Evaluate the log posterior of a batch given some neural network parameters for a regression task.

            Args:
                params (jnp.array): The parameters of the neural network
                batch (tuple of two jnp.array): (x,y) representing the data in the batch
                is_training (bool, optional): Whether this is run in training mode. Defaults to True.

            Returns:
                float: The log posterior
            """
            log_lik = log_likelihood_fct_regression(params, batch, is_training=True)
            log_prior = log_prior_fct(params)
            return log_lik + log_prior

    log_posterior_wgrad_fct = jax.jit(jax.value_and_grad(log_posterior_fct, argnums=0))

    epoch_steps = len(train_loader)
    total_steps = epoch_steps * nb_epochs

    def lr_schedule(step):
        """Schedule the learning rate

        Args:
            step (int): The number of steps performed

        Returns:
           float:The updated learning rate
        """
        t = step / total_steps
        return min_lr + 0.5 * (init_lr - min_lr) * (1 + jnp.cos(t * np.pi))

    # Initialize the optimizer
    if optimizer_args["optimizer_name"] == "adam":
        optimizer = optax.chain(
        optax.scale_by_adam(b1=optimizer_args["b1"], b2=optimizer_args["b2"], eps=1e-8),
        optax.scale_by_schedule(lr_schedule))

    elif optimizer_args["optimizer_name"] == "sgd":
        optimizer = optax.chain(
        optax.trace(decay=optimizer_args["momentum"], nesterov=optimizer_args["nesterov"]),
        optax.scale_by_schedule(lr_schedule))
    else:
        print("Unknown optimizer")

    opt_state = optimizer.init(params)

    @jax.jit
    def update_fct(batch, params, opt_state):
        """Perform an optimization step

        Args:
            batch (tuple of two jnp.array): (x,y) representing the data in the batch
            params (jnp.array): The parameters of the neural network
            opt_state (optax.OptState): The state of the optimizer

        Returns:
            tuple: (new_params, new_opt_state, loss)
                new_params (jnp.array): The updated neural network parameters
                new_opt_state (optax.OptState): The new optimizer state
                loss (float): The loss
        """

        # Extract data
        x, y = batch

        # Compute the loss and gradients
        loss, grad = log_posterior_wgrad_fct(params, (x, y))

        # Apply an optimization step
        updates, new_opt_state = optimizer.update(grad, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, loss

    def evaluate_fct(dataset, params):
        """Evaluate the model by computing the -log posterior

        Args:
            dataset (tf.data.Dataset): The dataset on which to perform the evaluation
            params (jnp.array): The parameters of the neural network to evaluate

        Returns:
            float: -log posterior
        """
        tot_val_loss = 0.
        for x, y in dataset:
            x, y = jnp.asarray(x), jnp.asarray(y)
            val_loss = -log_posterior_fct(params, (x, y), False)
            tot_val_loss += val_loss.item()

        return tot_val_loss / len(dataset)
    
    @jax.jit
    def get_accuracy_batch(batch, params):
        """Evaluate the model accuracy on a batch

        Args:
            batch (tuple of two jnp.array): (x,y) representing the data in the batch
            params (jnp.array): The parameters of the neural network

        Returns:
            float: The accuracy
        """

        # Extract data
        x, y = batch

        # Evaluate the neural network
        logits = model_apply(params, None, x, False)
        probs = jax.nn.softmax(logits, axis=1)

        # Compute the class with maximal probability
        preds = jnp.argmax(logits, axis=1)

        # Compute the accuracy
        accuracy = (preds == y).mean()
        return accuracy, probs

    def get_accuracy_fn(dataset, params):
        """Evaluate the model accuracy

        Args:
            dataset (tf.data.Dataset): The dataset on which to perform the evaluation
            params (jnp.array): The parameters of the neural network

        Returns:
            float: The accuracy
        """
        sum_accuracy = 0
        all_probs = []

        # Loop over the batches
        for x, y in dataset:
            x, y = jnp.asarray(x), jnp.asarray(y)

            # Compute batch accuracy
            batch_accuracy, batch_probs = get_accuracy_batch((x, y), params)
            sum_accuracy += batch_accuracy.item()
            all_probs.append(np.asarray(batch_probs))

        all_probs = jnp.concatenate(all_probs, axis=0)
        return sum_accuracy / len(dataset)

    best_val_loss = None
    best_weights = None

    train_loss = []
    val_loss = []

    epochs_made = 0
    epochs_since_improvement = 0

    # Perform for each epoch
    for epoch in range(nb_epochs):
        sum_loss = 0.

        # Loop over train set and update neural network for every batch
        for x, y in train_loader:
            x, y = jnp.asarray(x), jnp.asarray(y)
            params, opt_state, loss = update_fct((x, y), params, opt_state)
            sum_loss += loss

        # Log train loss
        avg_train_loss = -sum_loss/epoch_steps
        
        if not competition_mode:
            wandb.log({"train_loss": avg_train_loss.item()})
        
        # Evaluate the model
        if val_loader is not None:
            avg_val_loss = evaluate_fct(val_loader, params)

            if not competition_mode:
                accuracy = get_accuracy_fn(val_loader, params)
                wandb.log({"accuracy": accuracy})
                print("accuracy = {}".format(accuracy))

            print("Epoch {}, train loss: {} val loss: {}".format(epoch, avg_train_loss, avg_val_loss))
        
        else:
            avg_val_loss = avg_train_loss.item()
            print("Epoch {}, train loss: {}".format(epoch, avg_train_loss))

        # If max budget is specified and reached, stop the training 
        if max_budget is not None and epochs_made >= max_budget:
            print("max budget exceeded")
            return None, epochs_made

        epochs_made += 1

        # Update the weights to keep
        if keep_best_weights:
            if best_val_loss is None or avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_weights = params

        else:
            best_val_loss = avg_val_loss
            best_weights = params

        # Log evaluation metrics
        if not competition_mode:
            wandb.log({"best_val_loss": best_val_loss})
        train_loss.append(avg_train_loss)
        val_loss.append(avg_val_loss)

        # Update early stopping metrics and verify early stopping criterion
        if avg_val_loss > best_val_loss:
            epochs_since_improvement += 1
        else:
            epochs_since_improvement = 0
            
        if early_stopping and epochs_since_improvement >= early_stopping_epochs:
            break
    
    # Save losses and weights
    jnp.save(os.path.join(save_dir, "train_loss_{}.npy".format(model_index)), train_loss)
    jnp.save(os.path.join(save_dir, "val_loss_{}.npy".format(model_index)), val_loss)
    
    with open(os.path.join(save_dir, "weights_{}.pkl".format(model_index)), "wb") as fp:
        pickle.dump(best_weights, fp)

    return best_weights, epochs_made