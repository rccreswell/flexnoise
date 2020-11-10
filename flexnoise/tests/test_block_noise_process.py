"""Test the classes and methods contained in block_noise_process.py
"""

import flexnoise
import math
import numpy as np
import pints
import pints.toy
import unittest


class TestBlockNoiseProcess(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.kernel = flexnoise.kernels.LaplacianKernel
        cls.model = pints.toy.ConstantModel(1)
        cls.times = np.arange(1, 91)
        cls.data = np.hstack((
            np.random.normal(2.0, 0.1, 30),
            np.random.normal(2.0, 2.0, 30),
            np.random.normal(2.0, 0.1, 30))
        )
        cls.problem = pints.SingleOutputProblem(cls.model, cls.times, cls.data)
        cls.model_prior = pints.UniformLogPrior([0] * 1, [1e6] * 1)

    def test_posterior(self):
        block_noise_process = flexnoise.BlockNoiseProcess(
            self.problem, self.kernel, np.array([2.0]), self.model_prior)

        block_noise_process.posterior()

    def test_run_mcmc(self):
        block_noise_process = flexnoise.BlockNoiseProcess(
            self.problem, self.kernel, np.array([2.0]), self.model_prior)

        iters = 20
        model_params_chain, cov_chain = block_noise_process.run_mcmc(iters)

        self.assertEqual(len(model_params_chain), iters // 2)
        self.assertEqual(len(cov_chain), iters // 2)

    def test_log_poch(self):
        # Test the log Pochhammer function
        z = 4.0
        m = 3.0
        x = flexnoise.log_poch(z, m)
        self.assertAlmostEqual(x, math.log(4.0 * 5.0 * 6.0))

        z = 4.5
        m = 0.0
        x = flexnoise.log_poch(z, m)
        self.assertAlmostEqual(x, 0.0)

    def test_ppm_prior(self):
        # Test evaluating the PPM prior function
        sigma = 1e-3
        theta = 1e-3

        assignments = [0, 0, 0, 0, 0]
        p1 = flexnoise.ppm_prior(assignments, sigma, theta)

        assignments = [0, 1, 1, 2, 3]
        p2 = flexnoise.ppm_prior(assignments, sigma, theta)

        self.assertLess(p2, p1)

    def test_prior_K(self):
        # Test evaluating the marginal K prior
        sigma = 1e-3
        theta = 1e-3

        k = np.arange(11)
        n = 10

        prior_k = flexnoise.prior_K(k, n, sigma, theta)
        self.assertEqual(len(prior_k), len(k))
        self.assertAlmostEqual(prior_k.sum(), 1.0)
