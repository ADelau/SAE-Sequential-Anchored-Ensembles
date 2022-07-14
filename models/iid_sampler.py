import numpy as np
from scipy.stats import norm
import math

class IIDSampler():
	"""
	Sampler that draw samples independently
	"""
	def __init__(self, means, stds):
		"""Constructor

		Args:
			means (np.array): Means for each of the dimensions
			stds (np.array): Standard deviations for each of the dimensions
		"""
		self.means = means
		self.stds = stds

	def sample(self):
		"""Draw samples

		Returns:
			np.array: The samples
		"""
		
		# Draw samples from a normal distribution
		return np.random.normal(loc=self.means, scale=self.stds)

if __name__ == "__main__":
	"""
	Test the sampler
	"""

	from matplotlib import pyplot as plt
	NB_SAMPLES = 1000
	PRIOR_MEAN = 0.
	PRIOR_STD = math.sqrt(1.)
	NB_POINTS_PLOT = 1000
	BINS = 50

	sampler = IIDSampler(np.array([PRIOR_MEAN, PRIOR_MEAN]), np.array([PRIOR_STD, PRIOR_STD]))

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


