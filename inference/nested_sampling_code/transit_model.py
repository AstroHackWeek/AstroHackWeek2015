import numpy as np
import numpy.random as rng
import copy

# How many parameters are there?
num_params = 4

# Load the data
data = np.loadtxt('transit_data.txt')

# Some properties of the data
t_min, t_max = data[:,0].min(), data[:,0].max()
t_range = t_max - t_min
N = data.shape[0] # Number of data points

# Some idea of how big the Metropolis proposals should be
jump_sizes = np.array([200., 10., t_range, t_range])

def from_prior():
  """
  A function to generate parameter values from the prior.
  Returns a numpy array of parameter values.
  """
  A = -100. + 200.*rng.rand()
  b = 10.*rng.rand()
  tc = t_min + t_range*rng.rand()
  width = t_range*rng.rand()

  return np.array([A, b, tc, width])

def log_prior(params):
  """
  Evaluate the (log of the) prior distribution
  """
  A, b, tc, width = params[0], params[1], params[2], params[3]

  # Minus infinity, if out of bounds
  if A < -100. or A > 100.:
    return -np.Inf
  if b < 0. or b > 10.:
    return -np.Inf
  if tc < t_min or tc > t_max:
    return -np.Inf
  if width < 0. or width > t_range:
    return -np.Inf

  return 0.

def log_likelihood(params):
  """
  Evaluate the (log of the) likelihood function
  """
  # Rename the parameters
  A, b, tc, width = params[0], params[1], params[2], params[3]

  # First calculate the expected signal
  mu = A*np.ones(N)
  mu[np.abs(data[:,0] - tc) < 0.5*width] = A - b

  # Normal/gaussian distribution
  return -0.5*N*np.log(2.*np.pi) - np.sum(np.log(data[:,2])) \
            -0.5*np.sum((data[:,1] - mu)**2/data[:,2]**2)

def proposal(params):
  """
  Generate new values for the parameters, for the Metropolis algorithm.
  """
  # Copy the parameters
  new = copy.deepcopy(params)

  # Which one should we change?
  which = rng.randint(num_params)
  new[which] += jump_sizes[which]*10.**(1.5 - 6.*rng.rand())*rng.randn()
  return new


