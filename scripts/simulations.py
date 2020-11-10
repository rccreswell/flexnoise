"""Functions used for the figure generating scripts.
"""

import matplotlib.pyplot as plt
import numpy as np
import pints
import pints.toy
import warnings


def generate_time_series(model='logistic', noise='IID', n_times=200):
    """Generate a synthetic time series for testing.

    Currently it supports logistic growth with IID, AR(1), and blocked noise.

    Parameters
    ----------
    type : {''...}
        Which type of time-series to generate

    Returns
    -------
    np.ndarray
        time points
    np.ndarray
        noise free simulation
    np.ndarray
        noisy data
    """
    times = np.linspace(0, 100, n_times)

    if model == 'logistic':
        m = pints.toy.LogisticModel()
        real_parameters = [0.08, 50]
    else:
        raise ValueError('No known model {}'.format(model))

    values = m.simulate(real_parameters, times)

    if noise == 'IID':
        data = values + np.random.normal(0, 3, len(times))

    elif noise == 'AR1':
        data = values + pints.noise.ar1(0.8, 3, len(times))

    elif noise == 'multiplicative':
        noise = pints.noise.multiplicative_gaussian(2, 0.0075, values)
        data = values + noise

    elif noise == 'graphic':
        noise = pints.noise.multiplicative_gaussian(0.9, 0.5, values)
        data = values + noise

    elif noise == 'blocks':
        block_length = n_times // 5

        noise1 = np.random.normal(0, 3, block_length)
        noise2 = pints.noise.ar1(0.85, 3, block_length)
        noise3 = np.random.normal(0, 3, block_length)
        noise4 = np.random.normal(0, 30, block_length)
        noise5 = np.random.normal(0, 3, block_length)
        noise = np.concatenate((noise1, noise2, noise3, noise4, noise5))

        data = values + noise

    else:
        raise ValueError('No known noise process {}'.format(noise))

    return times, values, data


def run_pints(problem,
              likelihood,
              x0,
              num_mcmc_samples,
              num_chains=3,
              log_prior=None,
              likelihood_args=None,
              enforce_convergence=False,
              mcmc_method=None):
    """Perform infernce with Pints using a specified model and likelihood.

    Parameters
    ----------
    problem : pints.Problem
        Pints problem holding the times and data
    likelihood : pints.ProblemLogLikelihood
        Pints likelihood for the data
    x0 : array_like of float
        Starting point of model parameters.
    num_mcmc_samples : int
        Total number of MCMC samples.
    num_chains : int
        Number of separate MCMC chains.
    log_prior : pints.LogPrior
        Prior distribution on all parameters in the likelihood. If None, a
        uniform prior from 0 to 1e6 is chosen for all parameters.
    likelihood_args : list
        Any other arguments besides the pints problem which must be provided
        when instantiating the likelihood.
    enforce_convergence : bool
        Whether to raise an error if the chains have not converged. After
        finishing the MCMC chain, the Rhat value is calculated, and any value
        of Rhat greater than 1.05 is assumed to indicate lack of convergence.
    mcmc_method : str
        Name of any MCMC sampler implemented in Pints.

    Returns
    -------
    np.ndarray
        MCMC samples of posterior. One chain is provided with the first half
        discarded as burn-in.
    """
    if likelihood_args is None:
        log_likelihood = likelihood(problem)
    else:
        log_likelihood = likelihood(problem, *likelihood_args)

    # Get the number of parameters to infer = model params plus noise params
    num_params = len(x0)

    if log_prior is None:
        log_prior = pints.UniformLogPrior([0] * num_params, [1e6] * num_params)

    log_posterior = pints.LogPosterior(log_likelihood, log_prior)

    x0 = [np.array(x0), 1.1*np.array(x0), 0.9*np.array(x0)]

    # Run MCMC
    if mcmc_method is None:
        mcmc = pints.MCMCController(log_posterior, num_chains, x0)
    else:
        mcmc = pints.MCMCController(log_posterior, num_chains, x0,
                                    method=mcmc_method)
    mcmc.set_max_iterations(num_mcmc_samples)
    mcmc.set_log_to_screen(True)
    chains = mcmc.run()

    # Check convergence
    rs = pints.rhat(chains[:, num_mcmc_samples//2:, :])
    if max(rs) > 1.05:
        message = 'MCMC chains failed to converge, R={}'.format(str(rs))
        if enforce_convergence:
            pints.plot.trace(chains)
            plt.show()
            raise RuntimeError(message)
        else:
            warnings.warn(message)

    # Get one chain, discard first half burn in
    chain = chains[0][num_mcmc_samples//2:]

    return chain
