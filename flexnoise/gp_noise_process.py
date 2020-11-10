"""Smoothly varying noise process for time series.
"""

import flexnoise
import math
from multiprocessing import Pool
import numpy as np
import os
import pints
import random
import scipy.optimize
import warnings


class NonstatGPLogPrior(pints.ComposedLogPrior):
    r"""Builds a Gaussian process prior on each time varying kernel parameter.

    Uses the squared exponential kernel for covariance,

    .. math::
        k(x, y) = \alpha^2 e^{-(x - y)^2 / (2 \beta^2)}

    and a constant mean, mu.
    """
    def __init__(self, gp_times, num_gps, mu, alpha, beta):
        """
        For mu, alpha, and beta, either provide a single parameter which is
        used for all kernel parameters, or a list of parameters corresponding
        to the number of kernel parameters.

        Parameters
        ----------
        gp_times : np.ndarray
            The time points where Gaussian process is evaluated
        num_gps : int
            The number of time varying kernel parameters
        mu : float or list
            RBF mean
        alpha : float or list
            RBF scale
        beta : float or list
            RBF lengthscale
        """
        self._n_parameters = len(gp_times) * num_gps
        self._priors = []

        for i in range(num_gps):
            hyperparams = [0.0, 0.0, 0.0]
            for j, argument in enumerate((mu, alpha, beta)):
                try:
                    hyperparams[j] = argument[i]
                except (TypeError, IndexError):
                    hyperparams[j] = argument

            gp_prior_mean = hyperparams[0] * np.ones(len(gp_times))
            gp_prior_cov = hyperparams[1] ** 2 * \
                np.exp(-(gp_times - gp_times[:, np.newaxis]) ** 2
                       / (2 * hyperparams[2] ** 2))
            gp_prior_cov += 1e-3 * np.diag(np.ones(len(gp_times)))

            subprior = pints.MultivariateGaussianLogPrior(
                gp_prior_mean, gp_prior_cov)

            self._priors.append(subprior)


def optimize_worker(mail):
    """Worker function for optimization.

    This is a separate function for parallelization.

    Parameters
    ----------
    mail : tuple
        Contains the following items.
        posterior : pints.LogPosterior
            The posterior to optimize
        x0 : np.ndarray
            Initial condition
        options : dict
            Options to send to scipy L-BFGS-B
        seed : int
            Random seed

    Returns
    -------
    list
        First element is the optimal point, second is the function value at
        that point.
    """
    posterior, x0, options, seed = mail

    random.seed(seed)
    np.random.seed(seed // 2)

    def f(x):
        return -posterior(x)

    jac = None

    res = scipy.optimize.minimize(
        f,
        x0,
        jac=jac,
        method='L-BFGS-B',
        options=options
    )

    return [res.x, res.fun]


class GPNoiseProcess(flexnoise.NoiseProcess):
    def __init__(self,
                 problem,
                 kernel,
                 x0,
                 gp_times,
                 model_prior=None,
                 truncate=True):
        """
        Parameters
        ----------
        problem : pints.SingleOutputProblem
            The time series problem in Pints format
        kernel : type flexnoise.CovKernel
            Covariance function used within each block
        x0 : np.ndarray
            Starting condition for model parameters
        gp_times : np.ndarray
            Time points for the Gaussian process
        model_prior : pints.LogPDF, optional (None)
            Prior distribution over the model parameters. If not supplied, a
            uniform distribution from 1e-6 to 1e6 is used.
        truncate : bool, optional (True)
            Whether to pass the truncate small values option to the kernel
            parameter initializer. Best choice may depend on the kernel being
            used.
        """
        self.times = problem.times()
        self.data = problem.values()
        self.values = problem.evaluate(x0)
        self.problem = problem
        self.gp_times = gp_times
        self.kernel = kernel(None, gp_times)
        self.model_prior = model_prior
        self.truncate = truncate
        self.x0 = x0
        self.set_gp_hyperparameters(mu=0.0, alpha=1.0, beta_num_points=100)

    def _make_pints_posterior(self):
        """Rebuild the Pints posterior and save it.
        """
        # Build a uniform model prior if it is not supplied
        if self.model_prior is None:
            num_model_params = self.problem.n_parameters()
            model_prior = pints.UniformLogPrior([-1e6] * num_model_params,
                                                [1e6] * num_model_params)

        # Get the GP prior
        kernel_prior = NonstatGPLogPrior(
            self.gp_times,
            self.kernel.num_parameters() // len(self.gp_times),
            self.mu,
            self.alpha,
            self.beta)

        # Combine the two priors
        log_prior = pints.ComposedLogPrior(model_prior, kernel_prior)

        # Build the likelihood
        log_likelihood = flexnoise.KernelCovarianceLogLikelihood(
            self.problem, self.kernel)

        # Build the posterior
        log_posterior = pints.LogPosterior(log_likelihood, log_prior)

        self.posterior = log_posterior

    def set_gp_hyperparameters(self,
                               mu=None,
                               alpha=None,
                               beta=None,
                               beta_num_points=None,
                               beta_limit=0.01,
                               dt=None):
        """Set the hyperparameters of the Gaussian process.

        Each parameter is optional, and beta can be set either directly or
        using the beta_num_points (see GPNoiseProcess.set_gp_beta).

        Parameters
        ----------
        mu : float, optional (None)
            Mean
        alpha : float, optional (None)
            scale
        beta : float, optional (None)
            Lengthscale
        beta_num_points : int, optional (None)
            Number of points to use in the calculation of an appropriate beta
        beta_limit : float, optional (0.01)
            Small value when setting beta based on the covariance between
            data beta_num_points apart
        dt : int, optional (None)
            Spacing between time points. If not supplied, it will be taken from
            the beginning of the time series, and assumed constant.
        """
        if mu is not None:
            self.mu = mu

        if alpha is not None:
            self.alpha = alpha

        if beta is not None:
            self.beta = beta

        if beta_num_points is not None:
            if dt is None:
                dt = self.times[1] - self.times[0]
            self.set_gp_beta(beta_num_points, dt, limit=beta_limit)

        self._make_pints_posterior()

    def set_gp_beta(self, N, dt, limit=0.01):
        r"""Choose the variance hyperparameter of the RBF Gaussian process.

        This function returns the hyperparameter value beta which is given as
        the solution to the following equation

        ..math:
            limit = exp(-(N \Delta t)^2 / (2 \beta^2))

        The motivation of this equation is that under the prior, the covariance
        between two values of the Gaussian process N Dt apart is close to zero.

        Parameters
        ----------
        N : int
            Number of time points
        dt : float
            Spacing between each time point
        limit : float
            Small number giving the covariance between distant points

        Returns
        -------
        float
            The recommended prior hyperparameter value
        """
        beta = N * dt / math.sqrt(-2 * math.log(limit))
        self.beta = beta
        return beta

    def run_mcmc(self,
                 num_mcmc_samples,
                 num_chains,
                 iprint=True,
                 method=pints.PopulationMCMC,
                 enforce_convergence=False):
        """Run MCMC to obtain posterior samples.

        Parameters
        ----------
        num_mcmc_samples : int
            The total number of MCMC samples to run
        num_chains : int
            Number of separate MCMC chains to run
        iprint : bool, optional (True)
            Whether or not to print iteration number
        method : type, optional (pints.PopulationMCMC)
            Which MCMC method (pints.MCMCSampler) to use
        enforce_convergence : bool, optional (False)
            Whether to raise an error if the Rhat convergence statistic is less
            than 1.05.

        Returns
        -------
        np.ndarray
            MCMC chain, with shape (num_samples, num_parameters)
        """
        starting_points = self.get_initial_conditions(num_chains)

        mcmc = pints.MCMCController(
            self.posterior, num_chains, starting_points, method=method)
        mcmc.set_max_iterations(num_mcmc_samples)
        mcmc.set_log_to_screen(iprint)
        chains = mcmc.run()

        # Check convergence
        rs = pints.rhat(chains[:, num_mcmc_samples//2:, :])
        if max(rs) > 1.05:
            message = 'MCMC chains failed to converge, R={}'.format(str(rs))
            if enforce_convergence:
                raise RuntimeError(message)
            else:
                warnings.warn(message)

        # Get one chain and discard burn in
        chain = chains[0][num_mcmc_samples//2:]

        return chain

    def get_initial_conditions(self, num=1):
        """Get initial conditions for MCMC or optimization.

        This uses the initialize_parameters function of a covariance kernel.
        It supplies random values for the window size.

        The intial conditions also include the model parameter starting point
        at the front of each parameter vector.

        Parameters
        ----------
        num : int, optional (1)
            How many separate initial conditions to calculate

        Returns
        -------
        list of np.ndarray
            The initial conditions
        """
        starting_points = []
        for _ in range(num):
            # For each restart, get a random initial condition by picking a
            # random pair of window sizes for the init algorithm
            min_window_size = max(50, len(self.times) // 15)
            if min_window_size >= len(self.times):
                min_window_size = 3
            max_window_size = len(self.times) // 1.5

            window_size_lag = random.randint(min_window_size, max_window_size)
            window_size_std = random.randint(min_window_size, max_window_size)

            kernel_params = self.kernel.initialize_parameters(
                self.times,
                self.data - self.values,
                window_size_corr=window_size_lag,
                window_size_std=window_size_std,
                truncate_small_values=self.truncate
            )

            starting_point = np.concatenate((self.x0, kernel_params))
            starting_points.append(starting_point)

        return starting_points

    def run_optimize(self,
                     maxiter=100,
                     iprint=True,
                     num_restarts=10,
                     parallel=False):
        """Run optimisation to obtain MAP estimate.

        Parameters
        ----------
        maxiter : int, optional (100)
            Maximum number of iterations
        iprint : bool, optional (True)
            Whether to print the progress
        num_restarts : int, optional (10)
            Number of restarts
        parallel : bool, optional (False)
            Whether to run separate restarts in parallel

        Returns
        -------
        np.ndarray
            Parameters (model and kernel)
        """
        options = {'maxiter': maxiter,
                   'iprint': 1,
                   'maxls': 30,
                   'eps': 1e-6,
                   'ftol': 1e-7}

        if iprint is False:
            options['iprint'] = -1

        starting_points = self.get_initial_conditions(num_restarts)

        results = []
        final_objs = []

        if not parallel:
            for i in range(num_restarts):
                seed = random.randint(1, 2**32 - 1)
                mail = (self.posterior,
                        starting_points[i],
                        options,
                        seed)
                x, fx = optimize_worker(mail)
                results.append(x)
                final_objs.append(fx)

        else:
            pool = Pool(os.cpu_count() - 1)
            seed = random.randint(1, 2**32 - 1)
            mail = [(self.posterior,
                    starting_points[i],
                    options,
                    seed)
                    for i in range(num_restarts)]

            mail = pool.map(optimize_worker, mail)
            pool.close()
            pool.join()
            results = [x[0] for x in mail]
            final_objs = [x[1] for x in mail]

        # Return the best result
        return results[final_objs.index(min(final_objs))]
