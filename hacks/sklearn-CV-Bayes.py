from sklearn.base import BaseEstimator
import numpy as np

from straightline_utils import *
# from straightline_log_likelihood import straight_line_log_likelihood

# Define uniform prior limits, enforcing positivity in both parameters:
mlimits = [0.0, 2.0]
blimits = [0.0, 200.0]

class BayesMCMC(BaseEstimator):
    def __init__(self, nsteps=1000, mlower=0., mupper=2., blower=0., bupper=200.):
        self.nsteps = nsteps

    def straight_line_log_prior(self, m, b, mlimits, blimits):
        # Uniform in m:
        if (m < mlimits[0]) | (m > mlimits[1]):
            log_m_prior = -np.inf
        else:
            log_m_prior = np.log(1.0/(mlimits[1] - mlimits[0]))
        # Uniform in b:
        if (b < blimits[0]) | (b > blimits[1]):
            log_b_prior = -np.inf
        else:
            log_b_prior = np.log(1.0/(blimits[1] - blimits[0]))
            
        return log_m_prior + log_b_prior


    def straight_line_log_likelihood(self, x, y, sigmay, m, b):
        '''
        Returns the log-likelihood of drawing data values *y* at
        known values *x* given Gaussian measurement noise with standard
        deviation with known *sigmay*, where the "true" y values are
        *y_t = m * x + b*

        x: list of x coordinates
        y: list of y coordinates
        sigmay: list of y uncertainties
        m: scalar slope
        b: scalar line intercept

        Returns: scalar log likelihood
        '''

        return (np.sum(np.log(1./(np.sqrt(2.*np.pi) * sigmay))) +
                np.sum(-0.5 * (y - (m*x + b))**2 / sigmay**2))

    def straight_line_log_posterior(self, x, y, sigmay, m, b, mlimits, blimits):
        return (straight_line_log_likelihood(x,y,sigmay,m,b) +
                straight_line_log_prior(m,b,mlimits,blimits))


    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, shape = [n_samples, 2]
        y : array-like, shape = [n_samples]
            Target vector relative to x

        Returns
        -------
        self : object
            Returns self.
        """

        x, yerr = X[:,0], X[:,1]

        # Initial m, b, at center of prior:
        m = 0.5*(mlimits[0]+mlimits[1])
        b = 0.5*(blimits[0]+blimits[1])

        # Step sizes, 5% or 10% of the prior
        mstep = 0.05*(mlimits[1]-mlimits[0])
        bstep = 0.1*(blimits[1]-blimits[0])
                
        # We'll want to store the Markov chain as it evolves, since these are the samples we will
        # use in our inferences.
        self.chain = []
        self.probs = []
        naccept = 0
            
        print('Running Metropolis Sampler for', nsteps, 'steps...')

        # First point:
        L_old    = self.straight_line_log_likelihood(x, y, sigmay, m, b)
        p_old    = self.straight_line_log_prior(m, b, mlimits, blimits)
        logprob_old = L_old + p_old

        for i in range(nsteps):

            # Propose a step to a new point in parameter space:
            mnew = m + np.random.normal() * mstep
            bnew = b + np.random.normal() * bstep

            # Evaluate probabilities at the proposed point:
            L_new    = self.straight_line_log_likelihood(x, y, sigmay, mnew, bnew)
            p_new    = self.straight_line_log_prior(mnew, bnew, mlimits, blimits)
            logprob_new = L_new + p_new

            # Metropolis-Hastings acceptance criterion:
            if (np.exp(logprob_new - logprob_old) > np.random.uniform()):
                # Accept the proposed sample:
                m = mnew
                b = bnew
                L_old = L_new
                p_old = p_new
                logprob_old = logprob_new
                naccept += 1
            else:
                # Reject the proposed sample; m,b stay the same, and we append them
                # to the chain below.
                pass

            self.chain.append((b,m))
            self.probs.append((L_old,p_old))

        # All steps taken, report:    
            
        print('Acceptance fraction:', naccept/float(nsteps))
        self.acceptance = naccept / float(nsteps)


        return self


    def predict(self, X):
        # x_test, yerr_test = X[:,0], X[:,1]
        # compute the prediction from the model, for some m and b
        # return is free
        pass


    def score(self, X, y):
        # run MCMC
        self.fit(x, y, yerr)


        # Test statistics: functions of the data, not the parameters.

        # 1) Reduced chisq for the best fit model:
        def test_statistic(x,y,sigmay,b_ls,m_ls):
          return np.sum((y - m_ls*x - b_ls)**2.0/sigmay**2.0)/(len(y)-2)

        # 2) Reduced chisq for the best fit m=0 model:
        # def test_statistic(x,y,sigmay,dummy1,dummy2):
        #    return np.sum((y - np.mean(y))**2.0/sigmay**2.0)/(len(y)-1)

        # 3) Weighted mean y:
        # def test_statistic(x,y,sigmay,dummy1,dummy2):
        #    return np.sum(y/sigmay**2.0)/np.sum(1.0/sigmay**2.0)

        # 4) Variance of y:
        # def test_statistic(x,y,sigmay,dummy1,dummy2):
        #    return np.var(y)





(x,y,sigmay) = generate_data()

plot_yerr(x, y, sigmay)



b = BayesMCMC()
b.fit(x,y,sigmay)