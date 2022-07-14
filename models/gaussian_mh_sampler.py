import numpy as np
from scipy.stats import norm
import math

class GaussianMHSampler():
	"""
	Metropolis-Hastings sampler with Gaussian transitions. 
    Run a different Markov chain for each dimension.
	"""
	def __init__(self, means, stds, step_std):
		"""Constructor

		Args:
			means (np.array): Means for each of the dimensions
			stds (np.array): Standard deviations for each of the dimensions
            step_std (float): The standard deviation of the Gaussian transitions
		"""

		self.means = means
		self.stds = stds
		self.step_std = step_std
		self.current_sample = np.random.normal(loc=means, scale=stds)

	def sample(self):
		"""Grow the Markov chains from 1 sample

        Returns:
            np.array: The next samples
        """
		
		# Samples perturbations from a Gaussian
		noise = np.random.normal(loc=0., scale=self.step_std, size=self.means.shape)
		next_sample = self.current_sample + noise
		
		# Metropolis-Hastings step, since we use a gaussian proposal and hence q(x_{t+1}|x_t) = q(x_t|x_{t+1}), those terms can be dropped
		alpha = np.exp(norm(loc=self.means, scale=self.stds).logpdf(next_sample) - norm(loc=self.means, scale=self.stds).logpdf(self.current_sample))
		alpha = np.minimum(alpha, 1.)
		uniform_samples = np.random.uniform(low=0., high=1., size=alpha.shape)
		self.current_sample = np.where(uniform_samples <= alpha, next_sample, self.current_sample)

		return self.current_sample

if __name__ == "__main__":
	"""
	Test the sampler
	"""

	from matplotlib import pyplot as plt
	NB_SAMPLES = 1000
	PRIOR_MEAN = 0.
	PRIOR_STD = math.sqrt(1.)
	STEP_STD = 0.02
	NB_POINTS_PLOT = 1000
	BINS = 50

	sampler = GaussianMHSampler(np.array([PRIOR_MEAN, PRIOR_MEAN]), np.array([PRIOR_STD, PRIOR_STD]), STEP_STD)

	samples_1 = []
	samples_2 = []
	for i in range(NB_SAMPLES):
		if i%1000 == 0:
			print("iteration {}".format(i))
		results = sampler.sample()
		#print("results = {}".format(results))
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


