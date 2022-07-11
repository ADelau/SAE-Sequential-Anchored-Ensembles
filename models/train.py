import os
import math
import copy
from jax import numpy as jnp
import jax
import numpy as np
import optax
import pickle
import wandb


# Add gradient clipping
def train_model(train_loader, test_loader, model_apply, params, nb_epochs, train_set_size, init_lr, min_lr, save_dir, anchor, prior_variance, keep_best_weights, 
                optimizer_args, task, clip_gradient, competition_mode, model_index, clipping_norm=None, imbalance=False, imbalance_factor=0., imbalance_method=None, 
                max_gamma=0., early_stopping=False, early_stopping_epochs=1, max_budget=None):

    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass
        

    def log_likelihood_fct_classif(params, batch, is_training=True, imbalance_scale=None, gamma=None):
        x, y = batch
        logits = model_apply(params, None, x, is_training)
        num_classes = logits.shape[-1]
        labels = jax.nn.one_hot(y, num_classes)
        if imbalance_scale is None:
            softmax_xent = jnp.mean(jnp.sum(labels * jax.nn.log_softmax(logits), axis=1)) * train_set_size
        else:
            # focal loss
            if imbalance_method == "focal" or imbalance_method == "reweighting":
                tmp = labels * jax.nn.log_softmax(logits)
                softmax_xent = jnp.mean((1-tmp[:, 0])**gamma * tmp[:, 0] + imbalance_scale * (1-tmp[:, 1])**gamma * tmp[:, 1]) * train_set_size

            """
            elif imbalance_method == "reweighting":
                tmp = labels * jax.nn.log_softmax(logits)
                softmax_xent = jnp.mean(tmp[:,0] + imbalance_scale * tmp[:,1]) * train_set_size
            """
        
        return softmax_xent

    def log_likelihood_fct_regression(params, batch, is_training=True, imbalance_scale=None, gamma=None):
        x, y = batch
        predictions = model_apply(params, None, x, is_training)

        predictions_mean, predictions_std = jnp.split(predictions, [1], axis=-1)
        predictions_std = jax.nn.softplus(predictions_std)

        se = (predictions_mean - y)**2
        log_likelihood = (-0.5 * se / predictions_std**2 -
                          0.5 * jnp.log(predictions_std**2 * 2 * math.pi))
        log_likelihood = jnp.mean(log_likelihood) * train_set_size

        return log_likelihood


    def log_prior_fct(params):
        n_params = sum([p.size for p in jax.tree_leaves(params)])
        exp_term = sum([(-(jnp.ravel(p)-a)**2 / (2 * prior_variance)).sum() for p, a in zip(jax.tree_leaves(params), anchor)])
        """
        exp_term = sum(jax.tree_leaves(jax.tree_map(
            lambda p: (-p**2 / (2 * prior_variance)).sum(), params)))
        """
        norm_constant = -0.5 * n_params * jnp.log((2 * math.pi * prior_variance))
        return exp_term + norm_constant

    
    if task == "classification":
        @jax.jit
        def log_posterior_fct(params, batch, is_training=True, imbalance_scale=None, gamma=None):
            log_lik = log_likelihood_fct_classif(params, batch, is_training=True, imbalance_scale=imbalance_scale, gamma=gamma)
            log_prior = log_prior_fct(params)
            return log_lik + log_prior

    if task == "regression":
        @jax.jit
        def log_posterior_fct(params, batch, is_training=True, imbalance_scale=None, gamma=None):
            log_lik = log_likelihood_fct_regression(params, batch, is_training=True, imbalance_scale=imbalance_scale, gamma=gamma)
            log_prior = log_prior_fct(params)
            return log_lik + log_prior

    log_posterior_wgrad_fct = jax.jit(jax.value_and_grad(log_posterior_fct, argnums=0))

    epoch_steps = len(train_loader)
    total_steps = epoch_steps * nb_epochs

    def lr_schedule(step):
        t = step / total_steps
        return min_lr + 0.5 * (init_lr - min_lr) * (1 + jnp.cos(t * np.pi))

    if optimizer_args["optimizer_name"] == "adam":
        optimizer = optax.chain(
        #optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
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
    def update_fct(batch, params, opt_state, imbalance_scale=None, gamma=None):
        x, y = batch
        loss, grad = log_posterior_wgrad_fct(params, (x, y), imbalance_scale=imbalance_scale, gamma=gamma)
        updates, new_opt_state = optimizer.update(grad, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    def evaluate_fct(dataset, params):
        tot_test_loss = 0.
        for x, y in dataset:
            x, y = jnp.asarray(x), jnp.asarray(y)
            test_loss = -log_posterior_fct(params, (x, y), False)
            tot_test_loss += test_loss.item()

        return tot_test_loss / len(dataset)
    
    @jax.jit
    def get_accuracy_fn(batch, params):
        x, y = batch
        logits = model_apply(params, None, x, False)
        probs = jax.nn.softmax(logits, axis=1)
        preds = jnp.argmax(logits, axis=1)
        accuracy = (preds == y).mean()
        return accuracy, probs

    def evaluate_fct_tmp(dataset, params):
        sum_accuracy = 0
        all_probs = []
        for x, y in dataset:
            x, y = jnp.asarray(x), jnp.asarray(y)
            batch_accuracy, batch_probs = get_accuracy_fn((x, y), params)
            sum_accuracy += batch_accuracy.item()
            all_probs.append(np.asarray(batch_probs))
        all_probs = jnp.concatenate(all_probs, axis=0)
        return sum_accuracy / len(dataset), all_probs

    best_test_loss = None
    best_weights = None

    train_loss = []
    test_loss = []

    epochs_made = 0
    epochs_since_improvement = 0
    for epoch in range(nb_epochs):
        sum_loss = 0.

        if imbalance:
            if imbalance_method == "focal" or imbalance_method == "reweighting":
                imbalance_min_epoch = int(0.25*nb_epochs)
                imbalance_max_epoch = int(0.75*nb_epochs)
                if epoch <= imbalance_min_epoch:
                    imbalance_scale = imbalance_factor+1
                    gamma = max_gamma * (imbalance_max_epoch - epoch +1)/imbalance_max_epoch
                elif epoch <= imbalance_max_epoch:
                    imbalance_scale = imbalance_factor*((imbalance_max_epoch - epoch + 1)/(imbalance_max_epoch - imbalance_min_epoch)) + 1
                    gamma = max_gamma * (imbalance_max_epoch - epoch +1)/imbalance_max_epoch
                else:
                    imbalance_scale = None
                    gamma = None

            elif imbalance_method == "reweighting":
                imbalance_scale = imbalance_factor
        else:
            imbalance_scale = None
            gamma = None

        for x, y in train_loader:
            x, y = jnp.asarray(x), jnp.asarray(y)
            params, opt_state, loss = update_fct((x, y), params, opt_state, imbalance_scale=imbalance_scale, gamma=gamma)
            sum_loss += loss

        avg_train_loss = -sum_loss/epoch_steps
        
        if not competition_mode:
            wandb.log({"train_loss": avg_train_loss.item()})
        
        if test_loader is not None:
            avg_test_loss = evaluate_fct(test_loader, params)

            # Remove this for final submission
            if not competition_mode:
                accuracy, _ = evaluate_fct_tmp(test_loader, params)
                wandb.log({"accuracy": accuracy})
                print("accuracy = {}".format(accuracy))

            print("Epoch {}, train loss: {} test loss: {}".format(epoch, avg_train_loss, avg_test_loss))
        
        else:
            avg_test_loss = avg_train_loss.item()
            print("Epoch {}, train loss: {}".format(epoch, avg_train_loss))

        if max_budget is not None and epochs_made >= max_budget:
            print("max budget exceeded")
            return None, epochs_made

        epochs_made += 1

        if keep_best_weights:
            if best_test_loss is None or avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                best_weights = params

        else:
            best_test_loss = avg_test_loss
            best_weights = params

        if not competition_mode:
            wandb.log({"best_test_loss": best_test_loss})
        train_loss.append(avg_train_loss)
        test_loss.append(avg_test_loss)

        if avg_test_loss > best_test_loss:
            epochs_since_improvement += 1
        else:
            epochs_since_improvement = 0
            
        if early_stopping and epochs_since_improvement >= early_stopping_epochs:
            break

    jnp.save(os.path.join(save_dir, "train_loss_{}.npy".format(model_index)), train_loss)
    jnp.save(os.path.join(save_dir, "test_loss_{}.npy".format(model_index)), test_loss)
    
    with open(os.path.join(save_dir, "weights_{}.pkl".format(model_index)), "wb") as fp:
        pickle.dump(best_weights, fp)

    return best_weights, epochs_made