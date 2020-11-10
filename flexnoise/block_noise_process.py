"""Block covariance noise process for time series.
"""

import flexnoise
import math
import numpy as np
import random
import scipy.optimize
import scipy.special
from sksparse.cholmod import cholesky
from sksparse.cholmod import CholmodNotPositiveDefiniteError
import sys


class BlockNoiseProcess(flexnoise.NoiseProcess):
    def __init__(self,
                 problem,
                 kernel,
                 x0,
                 model_prior):
        """
        Parameters
        ----------
        problem : pints.SingleOutputProblem
            The time series problem in Pints format
        kernel : type flexnoise.CovKernel
            Covariance function used within each block
        x0 : np.ndarray
            Starting condition for model parameters
        model_prior : pints.LogPDF
            Prior distribution over the model parameters
        """

        self.problem = problem
        self.times = problem.times()
        self.data = problem.values()
        self._set_initial_blocks(5)
        self.kernel = kernel([])

        self.model_params = x0
        self.model_prior = model_prior

        self.model_prop_cov = np.diag(x0) * 0.1
        self.mu = x0.copy()
        self.log_a = 0

        self.sparse = True
        self.exceptions = 'inf'

        self.logl_prior = scipy.stats.norm(-4.0, 4.0)
        self.logs_prior = scipy.stats.norm(0.0, 4.0)

        self.hyper_sigma = 0.5
        self.hyper_theta = 1
        self.q = 0.1

    def _set_initial_blocks(self, num_blocks):
        """Initialize the block configuration.

        The initial condition is saved to self.assignments and
        self.kernel_params

        Parameters
        ----------
        num_blocks : int
            Starting number of blocks. They will be of equal size.
        """
        z = list(range(num_blocks)) * (len(self.data) // num_blocks)
        z.sort()
        self.assignments = z.copy()

        phi = []
        for _ in range(len(set(z))):
            phi.append([-5.0, 2.0])

        self.kernel_params = phi.copy()

    def posterior(self, blocks=[]):
        """Evaluate the log posterior.

        This method will refer to the assignments and parameters currently
        saved in the object.

        Parameters
        ----------
        blocks : list, optional ([])
            If a list of block indices are supplied, the posterior will only be
            evaluated for the time series data within those blocks. If [], it
            is evaluated for all blocks.

        Returns
        -------
        float
            Posterior
        """
        p = self.likelihood(blocks=blocks)
        p += ppm_prior(self.assignments, self.hyper_sigma, self.hyper_theta)

        num_clusters = len(set(self.assignments))

        phi = self.kernel_params
        for i in range(num_clusters):
            if i in blocks:
                p += self.logl_prior.logpdf(phi[i][0]) \
                     + self.logs_prior.logpdf(phi[i][1])

        return p

    def run_mcmc(self,
                 num_mcmc_samples,
                 iprint=True):
        """Run MCMC to generate samples from the posterior.

        Parameters
        ----------
        num_mcmc_samples : int
            The total number of MCMC samples to run
        iprint : bool, optional (True)
            Whether or not to print iteration number

        Returns
        -------
        list
            MCMC samples of model parameters
        list
            MCMC samples of covariance matrices (sparse format)
        """
        z_chain = []
        phi_chain = []
        theta_chain = []
        cov_chain = []
        hyper_sigma_chain = []
        hyper_theta_chain = []
        for iter in range(num_mcmc_samples):
            k = len(set(self.assignments))

            if iprint:
                print(iter, k)

            # Randomly choose either split or merge
            if (k == 1 or random.random() < self.q) and k < len(self.data):
                self._split_step()

            else:
                self._merge_step()

            # Recalculate k in case it changed in this iteration
            k = len(set(self.assignments))

            # Shuffle if possible
            if k > 1:
                self._shuffle_step()

            # Update parameters within each block
            self._update_kernel_params(iter)

            # Update model parameters
            self._update_model_params(iter)

            # Update block hyperparameters
            self._update_hyperparams()

            # Save values to the chain
            z_chain.append(self.assignments)
            phi_chain.append(self.kernel_params)
            theta_chain.append(self.model_params)
            cov = self.block_matrix()
            cov = scipy.sparse.block_diag(cov, format='csc')
            cov_chain.append(cov)
            hyper_sigma_chain.append(self.hyper_sigma)
            hyper_theta_chain.append(self.hyper_theta)

        return theta_chain[num_mcmc_samples//2:], \
            cov_chain[num_mcmc_samples//2:]

    def _split_step(self):
        """Propose a split, and accept or reject it.
        """
        k = len(set(self.assignments))
        z = self.assignments.copy()
        phi = self.kernel_params.copy()
        # Get all blocks with greater than 1 member
        splittable_blocks = \
            [block_idx for block_idx in set(z) if z.count(block_idx) > 1]

        # Choose a random one of those blocks
        j = random.choice(splittable_blocks)

        # Choose a random location within that block
        l = random.randint(1, z.count(j) - 1)

        # Build the proposed split vector
        j_time_idx = z.index(j)
        next_time_idx = j_time_idx + z.count(j)
        z_prop = z[:j_time_idx]
        for _ in range(l):
            z_prop.append(j)
        for _ in range(z.count(j)-l):
            z_prop.append(j+1)
        z_prop += [x + 1 for x in z[next_time_idx:]]

        # Propose a new value for the first newly split block
        old_params = phi[j]
        new_l = random.gauss(old_params[0], 4.0)
        new_s = random.gauss(old_params[1], 1.0)
        split_params = [new_l, new_s]

        phi_prop = phi.copy()
        phi_prop.insert(j, split_params)

        # Calculate acceptance ratio of the proposal
        p_old = self.posterior(blocks=[j, ])
        self.assignments = z_prop.copy()
        self.kernel_params = phi_prop.copy()
        p_prop = self.posterior(blocks=[j, j+1])
        q_prop = scipy.stats.norm.logpdf(new_l, old_params[0], 4.0)
        q_prop += scipy.stats.norm.logpdf(new_s, old_params[1], 1.0)
        log_alpha = p_prop - p_old - q_prop
        if k > 1:
            log_alpha += math.log(1-self.q) - math.log(self.q)
            ns = z.count(j)
            ngk = len(splittable_blocks)
            log_alpha += math.log(ngk * (ns - 1)) - math.log(k)

        elif k == 1:
            log_alpha += math.log(1-self.q) + math.log(len(self.data)-1)

        if math.log(random.random()) >= log_alpha:
            # Reject the proposal
            self.assignments = z.copy()
            self.kernel_params = phi.copy()

    def _merge_step(self):
        """Propose a merge, and accept or reject it.
        """
        k = len(set(self.assignments))
        z = self.assignments.copy()
        phi = self.kernel_params.copy()
        j = random.randint(0, k-2)

        # Build the proposed merged vector
        j_time_idx = z.index(j+1)
        z_prop = z[:j_time_idx]
        z_prop += [x - 1 for x in z[j_time_idx:]]

        # Build the proposed parameters by keeping the values in the second
        # block
        phi_prop = phi.copy()
        del phi_prop[j]

        # Calculate acceptance ratio of the proposal
        p_old = self.posterior(blocks=[j, j+1])
        self.assignments = z_prop.copy()
        self.kernel_params = phi_prop.copy()
        p_prop = self.posterior(blocks=[j, ])
        q_prop = scipy.stats.norm.logpdf(phi[j][0], phi[j+1][0], 4.0)
        q_prop += scipy.stats.norm.logpdf(phi[j][1], phi[j+1][1], 1.0)
        log_alpha = p_prop - p_old + q_prop
        if 1 < k < len(self.data):
            log_alpha += math.log(self.q) - math.log(1-self.q)
            ns = z.count(j)
            ns1 = z.count(j+1)
            ngk1 = len([block_idx for block_idx in set(z_prop)
                        if z_prop.count(block_idx) > 1])
            log_alpha += math.log(k-1) - math.log(ngk1 * (ns + ns1 - 1))

        elif k == len(self.data):
            log_alpha += math.log(self.q) + math.log(len(self.data)-1)

        if math.log(random.random()) >= log_alpha:
            # Reject the proposal
            self.assignments = z.copy()
            self.kernel_params = phi.copy()

    def _shuffle_step(self):
        """Propose a shuffle, and accept or reject it.
        """
        k = len(set(self.assignments))
        z = self.assignments.copy()

        for _ in range(3):
            # Perform the shuffle step
            # Choose a random block, which is not the last
            i = random.randint(0, k-2)
            i_time_index = z.index(i)
            next_time_index = i_time_index + z.count(i) + z.count(i+1)

            ni = z.count(i)
            ni1 = z.count(i+1)

            # Choose a new point for the change point somewhere within the two
            # blocks
            j = random.randint(0, ni + ni1 - 2)

            # Build the proposed shuffled assignments
            z_prop = z[:i_time_index]

            for _ in range(j + 1):
                z_prop.append(i)

            for _ in range(ni+ni1-j - 1):
                z_prop.append(i+1)

            z_prop += z[next_time_index:]

            p_old = self.posterior(blocks=[i, i+1])

            self.assignments = z_prop.copy()
            p_prop = self.posterior(blocks=[i, i+1])

            if math.log(random.random()) >= p_prop - p_old:
                # Reject the jump
                self.assignments = z.copy()

    def _update_kernel_params(self, iter):
        """Update kernel parameters within each block using MH steps.
        """
        z = self.assignments.copy()
        phi = self.kernel_params.copy()

        for i in range(len(set(z))):
            prop_cov = 0.01 * np.identity(len(phi[0]))
            accepted = 0
            rejected = 0
            sub_runs = 50 if (iter < 500 or iter % 10 == 0) else 5
            for subiter in range(sub_runs):
                phi = self.kernel_params.copy()
                x_old = phi[i]
                x_prop = scipy.stats.multivariate_normal.rvs(
                    mean=x_old, cov=prop_cov)
                phi_prop = phi.copy()
                phi_prop[i] = x_prop
                p_old = self.posterior(blocks=[i, ])
                self.kernel_params = phi_prop.copy()
                p_prop = self.posterior(blocks=[i, ])
                if math.log(random.random()) < p_prop - p_old:
                    accepted += 1
                else:
                    self.kernel_params = phi.copy()
                    rejected += 1
                if subiter % 15 == 0 and subiter != 0:
                    if accepted / (accepted + rejected) < 0.2:
                        prop_cov *= 0.5
                    else:
                        prop_cov *= 2.0
                    accepted = 0
                    rejected = 0

    def _update_model_params(self, iter):
        """Update model parameters using MH step.
        """
        theta = self.model_params.copy()

        if iter > 20:
            s = iter - 20
            gamma = (s + 1) ** -0.6

        theta_prop = scipy.stats.multivariate_normal.rvs(
            mean=theta, cov=math.exp(self.log_a) * self.model_prop_cov)
        theta_prop = np.reshape(theta_prop, theta.shape)

        l_old = self.likelihood()
        p_old = self.model_prior(theta)

        self.model_params = theta_prop.copy()
        l_prop = self.likelihood()
        p_prop = self.model_prior(theta_prop)

        if math.log(random.random()) < p_prop + l_prop - p_old - l_old:
            accepted = 1

        else:
            self.model_params = theta.copy()
            accepted = 0

        if iter > 20:
            theta = self.model_params.copy()
            self.model_prop_cov = (1.0-gamma) * self.model_prop_cov + gamma \
                * (theta - self.mu)[:, np.newaxis] \
                @ (theta - self.mu)[:, np.newaxis].T
            self.mu = (1.0-gamma) * self.mu + gamma * theta
            self.log_a += gamma * (accepted - 0.25)

    def _update_hyperparams(self):
        """Update partition model hyperparameters using MH step.
        """
        z = self.assignments.copy()
        hyper_sigma = self.hyper_sigma
        hyper_theta = self.hyper_theta
        for _ in range(20):
            hyper_sigma_prior = \
                lambda x: scipy.stats.beta.logpdf(x, 1, 1)
            a = 0.01
            b = 100
            hyper_theta_prior = \
                lambda x, s: scipy.stats.gamma.logpdf(x + s, a, scale=1/b)

            hyper_sigma_prop, hyper_theta_prop = \
                scipy.stats.multivariate_normal.rvs(
                    mean=[hyper_sigma, hyper_theta],
                    cov=np.identity(2) * 0.01)

            p_prop = hyper_sigma_prior(hyper_sigma_prop) \
                + hyper_theta_prior(hyper_theta_prop, hyper_sigma_prop)
            try:
                l_prop = ppm_prior(z, hyper_sigma_prop, hyper_theta_prop)
            except ValueError:
                l_prop = -np.inf

            l_old = ppm_prior(z, hyper_sigma, hyper_theta)
            p_old = hyper_sigma_prior(hyper_sigma) \
                + hyper_theta_prior(hyper_theta, hyper_sigma)

            if math.log(random.random()) < p_prop + l_prop - p_old - l_old:
                self.hyper_sigma = hyper_sigma_prop
                self.hyper_theta = hyper_theta_prop

    def block_matrix(self, mblocks=[]):
        """Build the covariance matrix given assignments and values.

        Parameters
        ----------
        mblocks : list of int, optional ([])
             Optional list of blocks. When supplied, only these blocks are
             used.

        Returns
        -------
        np.ndarray or list
            Array of the block covariance matrix in numpy format, or list of
            sparse matrices if self.sparse is set to True.
        """
        t = self.times
        blocks = []
        for regime in range(len(set(self.assignments))):
            if regime in mblocks or mblocks == []:
                length = self.assignments.count(regime)
                start = self.assignments.index(regime)
                regime_times = t[start:start+length]
                self.kernel.parameters = self.kernel_params[regime]
                if not self.sparse:
                    regime_block = self.kernel.get_matrix(regime_times)
                else:
                    regime_block = self.kernel.get_sparse_matrix(
                        regime_times, 1e-9)
                blocks.append(regime_block)

        if not self.sparse:
            cov = scipy.linalg.block_diag(*blocks)
            return cov

        else:
            return blocks

    def likelihood(self,
                   model_values=None,
                   blocks=[]):
        """Evaluate likelihood of data with the current block configuration.

        Parameters
        ----------
        model_values : np.ndarray, optional (None)
            The noise-free trajectory of the model simulated at the current
            model parameters. This is an optional argument, if it is provided
            then no further simulation from the model is necessary. In some
            cases, this may prevent unnecessary computations if the model
            trajectory is already known.
        blocks : list of int, optional ([])
             Optional list of blocks. When supplied, only these blocks are
             used. The output matrix will only have the covariance of those
             blocks.

        Returns
        -------
        float
            The log likelihood of the current block configuration and
            parameters.
        """
        z = np.array(self.assignments)
        x = self.data
        if blocks != []:
            keep = np.where((z >= min(blocks)) & (z <= max(blocks)))
            x = x[keep]

        cov = self.block_matrix(mblocks=blocks)

        if model_values is None:
            values = self.problem.evaluate(self.model_params)

        else:
            values = model_values

        if blocks != []:
            values = values[keep]

        try:
            if not self.sparse:
                ll = scipy.stats.multivariate_normal.logpdf(
                    x, mean=values, cov=cov)
                return ll

            else:
                l3 = 0
                pos = 0
                for sparse_cov in cov:
                    mu = values[pos:pos+sparse_cov.shape[0]]
                    xb = x[pos:pos+sparse_cov.shape[0]]
                    pos += sparse_cov.shape[0]
                    factor = cholesky(sparse_cov)
                    ld = factor.logdet()
                    k = len(mu)
                    l3 += -k / 2 * math.log(2 * math.pi) - 0.5 * ld \
                          - 0.5 * (xb - mu).T @ factor.solve_A(xb-mu)
                return l3

        except (RuntimeError, ValueError, CholmodNotPositiveDefiniteError):
            if self.exceptions == 'inf':
                return -math.inf
            else:
                print('likelihood failed ', sys.exc_info()[0])
                raise


def log_poch(z, m):
    """Return the logarithm of the rising factorial.

    The rising factorial is also known as the Pochhammer function.

    It is defined by z^(m) = z (z+1) (z+2) ... (z+m-1).

    If m=0, the result is zero.

    Parameters
    ----------
    z : float
    m : float

    Returns
    -------
    float
        Logarithm of rising factorial
    """
    return math.lgamma(z+m) - math.lgamma(z)


def find_prior_mean_theta(n, sigma=0.5, num_blocks=1.5):
    """Find the value of theta corresponding to a given sigma and mean blocks.
    """
    def f(theta):
        return -num_blocks + \
                scipy.special.poch(theta + sigma, n) / \
                (sigma * scipy.special.poch(theta + 1, n - 1)) - theta / sigma

    x = scipy.optimize.root_scalar(
        f, x0=sigma, bracket=[-sigma + 0.001, 1000*sigma]).root

    return x


def prior_K(k, n, sigma, theta):
    """Prior marginal probability of getting k blocks with n data points.

    Derivations and discussion are available in [1]_.

    Parameters
    ----------
    k : np.ndarray
        Values of the number of observed blocks. Must be between 1 and n.
    n : int
        Total number of time points
    sigma : float
        Hyperparameter 1
    theta : float
        Hyperparameter 2

    Returns
    -------
    np.ndarray
        Probability mass of the values of k

    References
    ----------
    .. [1] Martinez, Asael Fabian, and Ramses H. Mena. "On a nonparametric
           change point detection model in Markovian regimes." Bayesian
           Analysis 9.4 (2014): 823-858.
    """
    p = -scipy.special.loggamma(k + 1)
    prod1 = np.ones(len(k))
    for i, k_i in enumerate(k):
        for j in range(k_i - 1):
            prod1[i] *= theta + (j + 1) * sigma
    p += np.log(prod1)

    p -= log_poch(theta + 1, n - 1)
    p -= np.log(sigma ** k)

    def log_subtract_exp(x, y):
        return x + np.log1p(-np.exp(y-x))

    s = np.zeros(len(k))
    for i, k_i in enumerate(k):
        pos_terms = []
        neg_terms = []
        for j in range(k_i + 1):
            term = (-1) ** j * scipy.special.binom(k_i, j) * \
                 scipy.special.poch(-j * sigma, n)
            if term > 0:
                pos_terms.append(np.log(term))
            elif term < 0:
                neg_terms.append(np.log(-term))

        if len(pos_terms) > 0:
            log_sum_pos_terms = scipy.special.logsumexp(pos_terms)
        else:
            log_sum_pos_terms = -np.inf
        if len(neg_terms) > 0:
            log_sum_neg_terms = scipy.special.logsumexp(neg_terms)
        else:
            log_sum_neg_terms = -np.inf

        if np.abs(log_sum_pos_terms - log_sum_neg_terms) > 1e-13:
            log_s = log_subtract_exp(log_sum_pos_terms, log_sum_neg_terms)
        else:
            log_s = -math.inf

        s[i] = log_s

    pmf = p + s
    pmf = np.exp(pmf)
    return pmf / pmf.sum()


def ppm_prior(assignments, sigma, theta):
    """Calculate the log of the PPM prior for given block assignments.

    Parameters
    ----------
    assignments : list of int
        List giving the current block assignment for each time point
    sigma : int
        Discount prior hyperparameter
    theta : int
        Strength prior hyperparameter

    Returns
    -------
    float
        The log of the current value of the prior evaluated at the block
        assignments
    """
    n = len(assignments)  # number of time points
    k = len(set(assignments))  # number of blocks

    # Start calculating the terms of the prior in log space
    p = math.lgamma(n+1) - math.lgamma(k+1)

    prod1 = 0
    for i in range(k-1):
        prod1 += math.log(theta + (i+1) * sigma)
    p += prod1

    p -= log_poch(theta+1, n-1)

    prod2 = 0
    for j in range(k):
        n_j = assignments.count(j)
        prod2 += log_poch(1-sigma, n_j-1) - math.lgamma(n_j+1)

    p += prod2

    return p
