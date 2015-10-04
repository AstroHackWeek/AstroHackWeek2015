import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
import copy

# Set the seed
rng.seed(0)

# Import the model
from transit_model import from_prior, log_prior, log_likelihood, proposal,\
                              num_params

# Number of particles
N = 5

# Number of NS iterations
steps = 5*30

# MCMC steps per NS iteration
mcmc_steps = 10000

# Generate N particles from the prior
# and calculate their log likelihoods
particles = []
logp = np.empty(N)
logl = np.empty(N)
for i in range(0, N):
 x = from_prior()
 particles.append(x)
 logl[i] = log_likelihood(x)

# Storage for results
keep = np.empty((steps, num_params + 1))

plt.figure(figsize=(8, 8))
plt.ion()
plt.hold(False)

# Main NS loop
for i in range(0, steps):
  # Find worst particle
  worst = np.nonzero(logl == logl.min())[0]

  # Save its details
  keep[i, :-1] = particles[worst]
  keep[i, -1] = logl[worst]

  # Copy survivor
  if N > 1:
    which = rng.randint(N)
    while which == worst:
      which = rng.randint(N)
    particles[worst] = copy.deepcopy(particles[which])

  threshold = copy.deepcopy(logl[worst])

  # Evolve within likelihood constraint using Metropolis
  for j in range(0, mcmc_steps):
    new = proposal(particles[worst])
    logp_new = log_prior(new)
    # Only evaluate likelihood if prior prob isn't zero
    logl_new = -np.Inf
    if logp_new != -np.Inf:
      logl_new = log_likelihood(new)
    loga = logp_new - logp[worst]
    if loga > 0.:
      loga = 0.

    # Accept
    if logl_new >= threshold and rng.rand() <= np.exp(loga):
      particles[worst] = new
      logp[worst] = logp_new
      logl[worst] = logl_new

  # Use the deterministic approximation
  logX = -(np.arange(0, i+1) + 1.)/N

  plt.subplot(2,1,1)
  plt.plot(logX, keep[0:(i+1), -1], 'bo-')
  # Smart ylim
  temp = keep[0:(i+1), -1].copy()
  if len(temp) >= 2:
    np.sort(temp)
    plt.ylim([temp[0.2*len(temp)], temp[-1]])
  plt.ylabel('$\\log(L)$')

  plt.subplot(2,1,2)
  # Rough posterior weights
  logwt = logX.copy() + keep[0:(i+1), -1]
  wt = np.exp(logwt - logwt.max())
  plt.plot(logX, wt, 'bo-')
  plt.ylabel('Posterior weights (relative)')
  plt.xlabel('$\\log(X)$')
  plt.draw()

plt.ioff()
plt.show()

# Useful function
def logsumexp(values):
  biggest = np.max(values)
  x = values - biggest
  result = np.log(np.sum(np.exp(x))) + biggest
  return result

# Prior weights
logw = logX.copy()
# Normalise them
logw -= logsumexp(logw)

# Calculate marginal likelihood
logZ = logsumexp(logw + keep[:,-1])

# Normalised posterior weights
wt = wt/wt.sum()

effective_sample_size = int(np.exp(-np.sum(wt*np.log(wt + 1E-300))))

# Calculate information
H = np.sum(wt*(keep[:,-1] - logZ))

print('logZ = {logZ} +- {err}'.format(logZ=logZ, err=np.sqrt(H/N)))
print('Information = {H} nats'.format(H=H))
print('Effective Sample Size = {ess}'.format(ess=effective_sample_size))

posterior_samples = np.empty((effective_sample_size, keep.shape[1]))
k = 0
while True:
  # Choose one of the samples
  which = rng.randint(keep.shape[0])

  # Acceptance probability
  prob = wt[which]/wt.max()

  if rng.rand() <= prob:
    posterior_samples[k, :] = keep[which, :]
    k += 1

  if k >= effective_sample_size:
    break
# Save posterior samples
np.savetxt('keep.txt', posterior_samples)

