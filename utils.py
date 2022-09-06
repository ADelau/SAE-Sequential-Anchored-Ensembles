from metrics import *
import numpy as np
from jax import numpy as jnp
import jax

def evaluate_model(model, test_loader, true_probas, batch_size, test_size, task):
    """Evaluate the performance of a given model

    Args:
        model (SimpleModel, EnsembleModel, ...): A model such as found in models.py
        test_loader (tf.data.Dataset): The dataloader for the test set
        true_probas (np.array): The HMC probabilities (pass None if not available)
        batch_size (int): The batch size
        test_size (int): The size of the test set
        task (str): "classification" or "regression"

    Returns:
        if task is "classification"
            tuple: (pred_probas, agreement, total_variation, accuracy, likelihood)
                pred_probas (np.array): the predicted probabilities
                agreement (float): the agreement (0. if true_probas is None)
                total_variation (float): the total variation (0. if true_probas is None)
                accuracy (float): the accuracy
                likelihood (float): the likelihood of the test set given the model

        if test is "regreesion"
            tuple: (samples, w2, mse)
                samples (np.array): samples from the predicted distributions
                w2 (float): the W2 distance (0. if true_probas is None)
                mse (float): the mean squared error
    """

    if task == "classification":
        # Compute predicted probabilities
        pred_probas = model.predict(test_loader)

        # Compute the class with maximum probability
        preds = jnp.argmax(pred_probas, axis=1)
        num_classes = pred_probas.shape[-1]

        
        print("mean pred = {}".format(preds.mean()))

        # Evaluate accuracy and likelihood
        accuracy = 0.
        likelihood = 0.
        for i, batch in enumerate(test_loader):
            _, y = batch
            y = jnp.asarray(y)
            accuracy += jnp.sum(preds[i*batch_size: i*batch_size+len(y)] == y)
            labels = jax.nn.one_hot(y, num_classes)
            likelihood += jnp.sum(labels * jnp.log(pred_probas[i*batch_size: i*batch_size+len(y), :]))

        accuracy = accuracy/test_size

        # Evaluate agreemeent and total variation
        pred_probas = np.array(pred_probas)

        if true_probas is None:
            agreement = 0.
            total_variation = 0.
        else:
            agreement = compute_agreement(pred_probas, true_probas)
            total_variation = compute_total_variation(pred_probas, true_probas)

        return pred_probas, agreement, total_variation, accuracy, likelihood

    if task == "regression":
        # Draw samples from the predicted distributions
        samples = model.predict(test_loader)

        # Evaluate the mean squared error
        mse = 0.
        count = 0
        for i, batch in enumerate(test_loader):
            _, y = batch
            y = jnp.asarray(y)
            mse += jnp.sum((samples[i*batch_size: i*batch_size+len(y)] - y)**2)
            count += 1

        mse = mse/count
        samples = np.array(samples)

        # evaluate the W2 distance
        if true_probas is None:
            w2 = 0.
        else:
            w2 = w2_distance(samples, np.transpose(true_probas))

        return samples, w2, mse