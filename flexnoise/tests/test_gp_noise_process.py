"""Test the classes and methods contained in gp_noise_process.py
"""

import flexnoise
import math
import numpy as np
import pints
import pints.toy
import unittest


class TestGPNoiseProcess(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.kernel = flexnoise.kernels.GPLaplacianKernel
        cls.model = pints.toy.ConstantModel(1)
        cls.times = np.linspace(1.0, 4.0, 15)
        cls.data = np.random.normal(2.0, 1.0, 15)
        cls.gp_times = np.array([1.0, 2.5, 4.0])
        cls.problem = pints.SingleOutputProblem(cls.model, cls.times, cls.data)

    def test_construct(self):
        flexnoise.GPNoiseProcess(
            self.problem,
            self.kernel,
            [2.0],
            self.gp_times)

    def test_set_gp_hyperparameters(self):
        gp_noise_process = flexnoise.GPNoiseProcess(
            self.problem,
            self.kernel,
            [2.0],
            self.gp_times)

        gp_noise_process.set_gp_hyperparameters(
            mu=1.0,
            alpha=10.0,
            beta=2.0,
        )

        self.assertEqual(gp_noise_process.mu, 1.0)
        self.assertEqual(gp_noise_process.alpha, 10.0)
        self.assertEqual(gp_noise_process.beta, 2.0)

    def test_set_gp_beta(self):
        gp_noise_process = flexnoise.GPNoiseProcess(
            self.problem,
            self.kernel,
            [2.0],
            self.gp_times)

        dt = self.times[1] - self.times[0]
        limit = 0.01
        beta = gp_noise_process.set_gp_beta(100, dt, limit=0.01)
        expected = 100 * dt / math.sqrt(-2 * math.log(limit))

        self.assertAlmostEqual(beta, expected)

    def test_run_optimize(self):
        gp_noise_process = flexnoise.GPNoiseProcess(
            self.problem,
            self.kernel,
            [2.0],
            self.gp_times)

        y = gp_noise_process.run_optimize(num_restarts=3, iprint=False)
        self.assertEqual(
            len(y), self.problem.n_parameters() + 2 * len(self.gp_times))

    def test_run_optimize_parallel(self):
        gp_noise_process = flexnoise.GPNoiseProcess(
            self.problem,
            self.kernel,
            [2.0],
            self.gp_times)

        y = gp_noise_process.run_optimize(
            num_restarts=5, iprint=False, parallel=True)
        self.assertEqual(
            len(y), self.problem.n_parameters() + 2 * len(self.gp_times))

    def test_run_mcmc(self):
        gp_noise_process = flexnoise.GPNoiseProcess(
            self.problem,
            self.kernel,
            [2.0],
            self.gp_times)

        iters = 300
        chain = gp_noise_process.run_mcmc(
            iters,
            3,
            iprint=False)

        self.assertEqual(
            chain.shape,
            (iters // 2, self.problem.n_parameters() + 2 * len(self.gp_times)))
