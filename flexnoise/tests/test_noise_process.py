"""Test the classes in noise_process.py
"""

import flexnoise
import unittest


class TestNoiseProcess(unittest.TestCase):

    def test_run_mcmc(self):
        proc = flexnoise.NoiseProcess()

        with self.assertRaises(NotImplementedError):
            proc.run_mcmc()

    def test_run_optimise(self):
        proc = flexnoise.NoiseProcess()

        with self.assertRaises(NotImplementedError):
            proc.run_optimize()
