The implementation of the different available models can be found in the `models.py` file. The following models are implemented:
 * Simple neural networks
 * Deep ensembles and anchored ensembles (AE)
 * Sequential anchored ensembles (SAE)
 * Graphical anchored ensembles (an attempt to improve SAE not included in the paper)

## Simple neural network
A model composed of a simple neural network trained to minimize the log posterior. Let us denote by $D$ the dataset composed $N$ pairs $(x_i, y_i)$ and by $\theta$ the neural network weights. The neural network is trained to minimize

$-\log p(\theta|D) = -\log p(D|\theta) - \log p(\theta) + \log p(D)$,

which is equivalent to minimize

$- \log p(D|\theta) -\log p(\theta) \approx -\sum_{i=1, .., N} \log p(y_i|\theta, x_i) -\log p(\theta)$.

This is implemented in the class `SimpleModel`.

## Deep ensemble and anchored ensemble

A [Deep ensemble](https://arxiv.org/abs/1612.01474) consist of an ensemble of neural networks trained independently with the same loss function. Let us denote by $\theta_{i,..,n}$ the predictions are computed

$p(y|x, \theta_{i,..,n}) = \frac{1}{n} \sum_{i=1, .., n} p(y|x, \theta_i)$.

A [Anchored ensemble](https://arxiv.org/abs/1810.05546) aim to construct a deep ensemble that better matches the Bayesian posterior distribution. This is done by training the members of the ensemble with an anchored loss. An anchored loss uses an anchor $\theta_{\text{anc}}$ drawn from the prior $p(\theta)$. The anchored loss is then expressed

$-\sum_{i=1, .., N} \log p(y_i|\theta, x_i) -\log p_{\text{anc}}(\theta)$,

where $p_{\text{anc}}(\theta) = \mathcal{N}(\theta_{\text{anc}}, \Sigma_{\text{prior}})$.

This is implemented in the class `EnsembleModel`. Set the argument `anchored` to True to use anchored ensembles and to False to use deep ensembles.

## Sequential anchored ensemble
A [Sequential anchored ensemble](https://arxiv.org/abs/2201.00649) aim to efficiently construct an anchored ensemble by training the members sequentially. Every training procedure takes as starting point the optimum reached by the previous member with another anchor. In order to have consecutive anchors close to each other, the anchors $\theta_\text{anc}$ are sampled from an MCMC procedure that should eventually span the whole prior. Several MCMC samplers are provided. However, for the best performance, we recommend using the guided walk sampler. 

The different samplers available are
* **IID** The samples are drawn independently from the same distribution.

* **Gaussian Metropolis-Hastings** The samples are drawn using a Metropolis-Hastings procedure with a Gaussian proposal

* **Guided walk Metropolis-Hastings** The samples are drawn using a Guided-Walk Metropolis-Hastings procedure with a Gaussian proposal. The proposals always go in the same direction until a rejection occurs, switching the direction.

* **Hamiltonian Monte Carlo**
Proposals are drawn from Hamiltonian dynamics.

## Graphical anchored ensemble
Graphical anchored ensembles are an attempt to improve on sequential anchored ensembles that have not been included in the paper. It follows similar ideas of training the members sequentially for more efficient training. However, in this case, the anchors are not sampled from an MCMC procedure but i.i.d. from the prior distribution. The members are then trained in an order that minimizes the distance between consecutive anchors for faster training.
