import numpy as np
from scipy.stats import norm
import math
import tensorflow as tf
import jax.numpy as jnp
import jax


def gaussian_energy(x, means=0., sigmas=1.):
    """Compute the energy function

    Args:
        x (np.array or float): The position(s)
        means (np.array or float, optional): The mean(s). Defaults to 0..
        sigmas (np.array or float, optional): The standard deviation(s). Defaults to 1.

    Returns:
        np.array or float: The associated energ(y/ies)
    """
    
    n_params = len(x)
    exp_term = -(x-means)**2 / (2 * sigmas**2)
    norm_constant = -0.5 * n_params * np.log((2 * math.pi * sigmas**2))
    return -(exp_term + norm_constant)

def gaussian_energy_grad(x, means=0., sigmas=1):
    """Compute the gradient of the energy function

    Args:
        x (np.array or float): The position(s)
        means (np.array or float, optional): The mean(s). Defaults to 0..
        sigmas (np.array or float, optional): The standard deviation(s). Defaults to 1.

    Returns:
        np.array or float: The associated energy gradient(s)
    """

    exp_term = -2*(x-means) / (2 * sigmas**2)
    return -exp_term

def kinetic_energy(velocity):
    """Compute the kinetic energy

    Args:
        velocity (np.array): The velocities

    Returns:
        np.array: The associated kinetic energies
    """

    return 0.5 * velocity**2

def hamiltonian(position, velocity, energy):
    """Compute the Hamiltonians

    Args:
        position (np.array): The positions
        velocity (np.array): The velocities
        energy (function): The energy function

    Returns:
        np.array: The associated Hamiltonians
    """

    return energy(position) + kinetic_energy(velocity)

def leapfrog_step(x0, v0, energy, grad_energy, step_size, num_steps):
    """Perform several leapfrog steps

    Args:
        x0 (np.array): The initial positions
        v0 (np.array): The initial velocities
        energy (function): The energy function
        grad_energy (function):The gradient of the energy function
        step_size (float): The size of a leapfrog step
        num_steps (int): Number of steps performed

    Returns:
        tuple: (x, v, loss)
            x (np.array): The new positions
            v (np.array): The new velocities
    """
    
    gradient = grad_energy(x0)
    v = v0 - 0.5 * step_size * gradient
    x = x0 + step_size * v

    for i in range(num_steps):
        gradient = grad_energy(x)
        v = v -  step_size * gradient
        x = x + step_size * v

    v = v - 0.5 * step_size * grad_energy(x)

    return x, v

def hmc(initial_x, step_size, num_steps, energy, grad_energy):
    """Grow the Markov chains from 1 sample

    Args:
        initial_x (np.array): Current samples in the Markov chains
        step_size (float): The size of a leapfrog step
        num_steps (int): Number of steps performed between each Metropolis-Hastings correction
        energy (function): The energy function
        grad_energy (function):The gradient of the energy function

    Returns:
        np.array: Next samples in the Markov chains
    """

    v0 = np.random.normal(initial_x.shape)

    # Perform leap frog steps
    x, v = leapfrog_step(initial_x,
                      v0, 
                      step_size=step_size, 
                      num_steps=num_steps, 
                      energy=energy,
                      grad_energy=grad_energy)

    # Metropolis-Hastings correction
    orig = hamiltonian(initial_x, v0, energy)

    current = hamiltonian(x, v, energy)

    alpha = np.exp(orig - current)
    alpha = np.minimum(alpha, 1.)
    
    uniform_samples = np.random.uniform(low=0., high=1., size=alpha.shape)
    return np.where(uniform_samples <= alpha, x, initial_x)


class HamiltonianSampler():
    """
	Hamiltonian Monte Carlo (HMC) sampler. 
    Draw samples using MCMC and Hamiltonian dynamics. 
    Run a different Markov chain for each dimension.
	"""
    def __init__(self, means, stds, step_size, num_steps):
        """Constructor

		Args:
			means (np.array): Means for each of the dimensions
			stds (np.array): Standard deviations for each of the dimensions
            step_size (float): The size of a leapfrog step
            num_steps (int): Number of steps performed between each Metropolis-Hastings correction
		"""

        self.means = means
        self.stds = stds
        self.step_size = step_size
        self.num_steps = num_steps

        self.current_sample = np.random.normal(loc=np.zeros(stds.shape), scale=stds)
        self.energy = lambda x: gaussian_energy(x, means=means, sigmas=stds)
        self.grad_energy = lambda x: gaussian_energy_grad(x, means=means, sigmas=stds)

    def sample(self):
        """Grow the Markov chains from 1 sample

        Returns:
            np.array: The next samples
        """
        
        self.current_sample = hmc(self.current_sample, self.step_size, self.num_steps, self.energy, self.grad_energy)

        return np.array(self.current_sample)

if __name__ == "__main__":
    """
	Test the sampler
	"""

    from matplotlib import pyplot as plt
    NB_SAMPLES = 1000
    PRIOR_MEAN = 0.
    PRIOR_STD = math.sqrt(1.)
    STEP_SIZE = 0.04
    NUM_STEPS = 70
    NB_POINTS_PLOT = 1000
    BINS = 50

    sampler = HamiltonianSampler(np.array([PRIOR_MEAN, PRIOR_MEAN]), np.array([PRIOR_STD, PRIOR_STD]), STEP_SIZE, NUM_STEPS)

    samples_1 = []
    samples_2 = []
    for i in range(NB_SAMPLES):
        if i%1000== 0:
            print("iteration {}".format(i))
        results = sampler.sample()

        samples_1.append(results[0])
        samples_2.append(results[1])

    def plot_samples(samples):
        plot_values = np.linspace(min(samples), max(samples), NB_POINTS_PLOT)
        densities = norm.pdf(plot_values, loc=PRIOR_MEAN, scale=PRIOR_STD)

        plt.hist(samples, bins=BINS, density=True)
        plt.plot(plot_values, densities)
        plt.show()

    plot_samples(samples_1)
    plot_samples(samples_2)


