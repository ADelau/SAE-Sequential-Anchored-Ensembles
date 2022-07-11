from metrics import *
import numpy as np
from jax import numpy as jnp
import jax

def evaluate_model(model, test_loader, true_probas, batch_size, test_size, task):

    if task == "classification":
        pred_probas = model.predict(test_loader, batch_size)

        preds = jnp.argmax(pred_probas, axis=1)
        num_classes = pred_probas.shape[-1]

        
        print("mean pred = {}".format(preds.mean()))

        accuracy = 0.
        likelihood = 0.
        for i, batch in enumerate(test_loader):
            _, y = batch
            y = jnp.asarray(y)
            accuracy += jnp.sum(preds[i*batch_size: i*batch_size+len(y)] == y)
            labels = jax.nn.one_hot(y, num_classes)
            likelihood += jnp.sum(labels * jnp.log(pred_probas[i*batch_size: i*batch_size+len(y), :]))

        accuracy = accuracy/test_size

        pred_probas = np.array(pred_probas)

        if true_probas is None:
            agreement = 0.
            total_variation = 0.
        else:
            agreement = compute_agreement(pred_probas, true_probas)
            total_variation = compute_total_variation(pred_probas, true_probas)

        return pred_probas, agreement, total_variation, accuracy, likelihood

    if task == "regression":
        samples = model.predict(test_loader, batch_size)

        mse = 0.
        count = 0
        for i, batch in enumerate(test_loader):
            _, y = batch
            y = jnp.asarray(y)
            mse += jnp.sum((samples[i*batch_size: i*batch_size+len(y)] - y)**2)
            count += 1

        mse = mse/count
        samples = np.array(samples)

        if true_probas is None:
            w2 = 0.
        else:
            w2 = w2_distance(samples, np.transpose(true_probas))

        return samples, w2, mse