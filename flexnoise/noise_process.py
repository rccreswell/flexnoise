"""Noise processes for time series
"""


class NoiseProcess:
    """Base class for noise processes.
    """

    def __init__(self):
        pass

    def run_mcmc(self):
        """Run MCMC to generate samples from the posterior.
        """
        raise NotImplementedError

    def run_optimize(self):
        """Run optimization to find a MAP estimate.
        """
        raise NotImplementedError
