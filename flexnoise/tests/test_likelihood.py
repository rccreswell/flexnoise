"""Test the classes and methods contained in likelihood.py.
"""

import flexnoise
import math
import numpy as np
import pints
import pints.toy
import scipy.sparse
import scipy.stats
import unittest


class TestCovarianceLogLikelihood(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = pints.toy.ConstantModel(1)
        cls.times = np.array([1.0, 2.0, 3.0, 4.0])
        cls.data = np.array([2.0, 2.5, 1.5, 2.0])
        cls.problem = pints.SingleOutputProblem(cls.model, cls.times, cls.data)

    def test_call_sparse_matrix(self):
        # Test evaluating the likelihood with a sparse matrix
        cov = np.identity(4) * 1.5
        sparse_cov = scipy.sparse.csc_matrix(cov)

        ll = flexnoise.CovarianceLogLikelihood(self.problem, sparse_cov)([2.0])

        expected = scipy.stats.norm.logpdf(
            self.data, loc=2.0, scale=math.sqrt(1.5))

        assert np.allclose(ll, expected.sum())

    def test_call_dense_matrix(self):
        # Test evaluating the likelihood with a dense matrix
        cov = np.identity(4) * 1.5

        ll = flexnoise.CovarianceLogLikelihood(self.problem, cov)([2.0])

        expected = scipy.stats.norm.logpdf(
            self.data, loc=2.0, scale=math.sqrt(1.5))

        assert np.allclose(ll, expected.sum())

    def test_exceptions(self):
        # Test the exceptions with faulty inputs

        # Make a non positive definite matrix
        cov = np.ones((4, 4))

        ll = flexnoise.CovarianceLogLikelihood(
            self.problem, cov, exceptions='strict')

        with self.assertRaises(np.linalg.LinAlgError):
            x = ll([2.0])

        ll = flexnoise.CovarianceLogLikelihood(
            self.problem, cov, exceptions='inf')

        x = ll([2.0])
        assert np.isneginf(x)


class SimpleKernel(flexnoise.kernels.CovKernel):
    """Diagonal kernel for testing.
    """
    def __init__(self, s):
        self.parameters = s
        self.sparse_method = 'cutoff'

    def __call__(self, x, y):
        p = self.parameters[0]
        return (x == y) * p ** 2

    def get_matrix(self, t):
        p = self.parameters[0]
        m = np.identity(len(t)) * p ** 2
        return m

    def num_parameters(self):
        return 1


class TestKernelCovarianceLogLikelihood(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = pints.toy.ConstantModel(1)
        cls.times = np.array([1.0, 2.0, 3.0, 4.0])
        cls.data = np.array([2.0, 2.5, 1.5, 2.0])
        cls.problem = pints.SingleOutputProblem(cls.model, cls.times, cls.data)
        cls.kernel = SimpleKernel([])

    def test_call_sparse_matrix(self):
        # Test evaluating the likelihood with a sparse matrix

        ll = flexnoise.KernelCovarianceLogLikelihood(self.problem, self.kernel)
        x = ll([2.1, 3.7])

        expected = scipy.stats.norm.logpdf(
            self.data, loc=2.1, scale=3.7)

        assert np.allclose(x, expected.sum())

    def test_call_dense_matrix(self):
        # Test evaluating the likelihood with a dense matrix

        ll = flexnoise.KernelCovarianceLogLikelihood(
            self.problem, self.kernel, use_sparse=False)
        x = ll([2.1, 3.7])

        expected = scipy.stats.norm.logpdf(
            self.data, loc=2.1, scale=3.7)

        assert np.allclose(x, expected.sum())


class TestKnownBlocksCovarianceLogLikelihood(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = pints.toy.ConstantModel(1)
        cls.times = np.array([1.0, 2.0, 3.0, 4.0])
        cls.data = np.array([2.0, 2.5, 1.5, 2.0])
        cls.problem = pints.SingleOutputProblem(cls.model, cls.times, cls.data)
        cls.kernel = SimpleKernel([])

    def test_call_sparse_matrix(self):
        # Test evaluating the likelihood with a sparse matrix

        blocks = [[0, 1], [1, 2]]

        ll = flexnoise.KnownBlocksCovarianceLogLikelihood(
            self.problem, self.kernel, blocks)
        x = ll([2.1, 3.7, 4.2])

        expected = \
            scipy.stats.norm.logpdf(self.data[:2], loc=2.1, scale=3.7).sum() \
            + scipy.stats.norm.logpdf(self.data[2:], loc=2.1, scale=4.2).sum()

        assert np.allclose(x, expected)

    def test_call_dense_matrix(self):
        # Test evaluating the likelihood with a dense matrix

        blocks = [[0, 1], [1, 2]]

        ll = flexnoise.KnownBlocksCovarianceLogLikelihood(
            self.problem, self.kernel, blocks, use_sparse=False)
        x = ll([2.1, 3.7, 4.2])

        expected = \
            scipy.stats.norm.logpdf(self.data[:2], loc=2.1, scale=3.7).sum() \
            + scipy.stats.norm.logpdf(self.data[2:], loc=2.1, scale=4.2).sum()

        assert np.allclose(x, expected)
