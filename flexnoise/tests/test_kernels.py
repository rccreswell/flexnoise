"""Test the classes and methods contained in kernels.py
"""

import flexnoise
import math
import numpy as np
import unittest


class TestLaplacianKernel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.K = flexnoise.kernels.LaplacianKernel([])

    def test_call(self):
        # Test that the kernel returns the correct numbers when called
        self.K.parameters = [1.5, 2.5]
        x = self.K(3.3, 5.7)
        expected1 = \
            math.exp(2.5)**2 * math.exp(-abs(3.3 - 5.7) / math.exp(1.5))
        assert np.allclose(x, expected1)

        x = self.K(8.9, 0.9)
        expected2 = \
            math.exp(2.5)**2 * math.exp(-abs(8.9 - 0.9) / math.exp(1.5))
        assert np.allclose(x, expected2)

        # Test with a vector input
        x = self.K(np.array([3.3, 8.9]), np.array([5.7, 0.9]))
        expected_array = np.array([expected1, expected2])
        assert np.allclose(x, expected_array)

    def test_matrix(self):
        # Test getting a matrix
        self.K.parameters = [1.5, 2.5]
        t = np.linspace(0, 10, 10)
        m = self.K.get_matrix(t)

        expected = np.zeros((len(t), len(t)))
        for i, ti in enumerate(t):
            for j, tj in enumerate(t):
                expected[i, j] = \
                    math.exp(2.5)**2 * math.exp(-abs(ti - tj) / math.exp(1.5))

        assert np.allclose(m, expected)

    def test_num_parameters(self):
        # Test the function that returns number of parameters
        assert self.K.num_parameters() == 2

    def test_sparse_matrix_diagonal(self):
        # Test getting a sparse matrix using diagonal method
        self.K.parameters = [0.1, 2.5]
        self.K.sparse_method = 'diagonal'
        t = np.linspace(0, 50, 10)

        m_sparse = self.K.get_sparse_matrix(t, 1e-10)
        m = m_sparse.toarray()

        expected = np.zeros((len(t), len(t)))
        for i, ti in enumerate(t):
            for j, tj in enumerate(t):
                expected[i, j] = \
                    math.exp(2.5)**2 * math.exp(-abs(ti - tj) / math.exp(0.1))

        assert np.allclose(m, expected)

    def test_sparse_matrix_cutoff(self):
        # Test getting a sparse matrix using cutoff method
        self.K.parameters = [0.1, 2.5]
        self.K.sparse_method = 'cutoff'
        t = np.linspace(0, 50, 10)

        m_sparse = self.K.get_sparse_matrix(t, 1e-10)
        m = m_sparse.toarray()

        expected = np.zeros((len(t), len(t)))
        for i, ti in enumerate(t):
            for j, tj in enumerate(t):
                expected[i, j] = \
                    math.exp(2.5)**2 * math.exp(-abs(ti - tj) / math.exp(0.1))

        assert np.allclose(m, expected)


class TestCovKernel(unittest.TestCase):

    def test_methods(self):
        kernel = flexnoise.kernels.CovKernel()

        with self.assertRaises(NotImplementedError):
            kernel(1.0, 2.0)

        with self.assertRaises(NotImplementedError):
            kernel.get_matrix([1.0, 2.0])

        with self.assertRaises(NotImplementedError):
            kernel.num_parameters()

        with self.assertRaises(NotImplementedError):
            kernel.initialize_parameters()


class TestGPLaplacianKernel(unittest.TestCase):

    def test_call(self):
        # Test that the kernel returns the correct numbers when called
        K = flexnoise.kernels.GPLaplacianKernel(
            np.array([1.0, 1.5, 1.75, 10.0, 5.0, 4.0]),
            np.array([1.0, 2.0, 3.0])
        )

        x = K(1.0, 2.0)
        expected1 = math.exp(10.0) * math.exp(5.0) \
            * np.sqrt(2 * math.exp(1.0) * math.exp(1.5)
                      / (math.exp(1.0)**2 + math.exp(1.5)**2)) \
            * math.exp(-abs(1.0-2.0)
                       / np.sqrt(math.exp(1.0)**2 + math.exp(1.5)**2))

        assert np.allclose(x, expected1)

        x = K(1.0, 3.0)
        expected2 = math.exp(10.0) * math.exp(4.0) \
            * np.sqrt(2 * math.exp(1.0) * math.exp(1.75)
                      / (math.exp(1.0)**2 + math.exp(1.75)**2)) \
            * math.exp(-abs(1.0-3.0)
                       / np.sqrt(math.exp(1.0)**2 + math.exp(1.75)**2))

        assert np.allclose(x, expected2)

        # Test with a vector input
        x = K(np.array([1.0, 1.0]), np.array([2.0, 3.0]))
        expected = np.array([expected1, expected2])

        assert np.allclose(x, expected)

    def test_matrix(self):
        # Test getting a matrix
        K = flexnoise.kernels.GPLaplacianKernel(
            np.array([1.0, 1.0, 1.2, 0.1, 0.2, 0.4]),
            np.array([1.0, 2.0, 3.0])
        )

        m = K.get_matrix(np.array([1.0, 2.0, 3.0]))

        s = [0.1, 0.2, 0.4]
        L = [1.0, 1.0, 1.2]
        t = [1.0, 2.0, 3.0]

        expected = np.zeros((3, 3))
        for i, ti in enumerate(t):
            for j, tj in enumerate(t):
                expected[i, j] = math.exp(s[i]) * math.exp(s[j]) \
                    * np.sqrt(2 * math.exp(L[i]) * math.exp(L[j])
                              / (math.exp(L[i])**2 + math.exp(L[j])**2)) \
                    * math.exp(-abs(ti - tj)
                               / np.sqrt(math.exp(L[i])**2
                                         + math.exp(L[j])**2))

        assert np.allclose(m, expected)

    def test_transforms(self):
        # Test with a transform matrix

        transform1 = np.identity(3) * 0.5
        transform2 = np.identity(3) * 2.0
        K = flexnoise.kernels.GPLaplacianKernel(
            np.array([1.0, 1.0, 1.2, 0.1, 0.2, 0.4]),
            np.array([1.0, 2.0, 3.0]),
            transforms=[transform1, transform2],
        )

        m = K.get_matrix(np.array([1.0, 2.0, 3.0]))

        s = np.array([0.1, 0.2, 0.4]) * 2.0
        L = np.array([1.0, 1.0, 1.2]) * 0.5
        t = [1.0, 2.0, 3.0]

        expected = np.zeros((3, 3))
        for i, ti in enumerate(t):
            for j, tj in enumerate(t):
                expected[i, j] = math.exp(s[i]) * math.exp(s[j]) \
                    * np.sqrt(2 * math.exp(L[i]) * math.exp(L[j])
                              / (math.exp(L[i])**2 + math.exp(L[j])**2)) \
                    * math.exp(-abs(ti - tj)
                               / np.sqrt(math.exp(L[i])**2
                                         + math.exp(L[j])**2))

        assert np.allclose(m, expected)

        # Make independent residuals that increase in magnitude over time
        residuals = np.random.normal(0, 1, 1000) * np.linspace(1, 7, 1000)
        times = np.linspace(0, 10, 1000)

        K.initialize_parameters(times, residuals)

    def test_num_parameters(self):
        # Test the function that returns number of parameters
        K = flexnoise.kernels.GPLaplacianKernel(
            np.array([1.0, 1.5, 1.75, 0.1, 0.5, 0.4]),
            np.array([1.0, 2.0, 3.0])
        )
        assert K.num_parameters() == 6

    def test_sparse_matrix_diagonal(self):
        # Test getting a sparse matrix using diagonal method

        t = np.array([1.0, 2.0, 3.0])

        K = flexnoise.kernels.GPLaplacianKernel(
            np.array([1.0, 1.0, 1.2, 0.1, 0.2, 0.4]), t)

        K.sparse_method = 'diagonal'

        m_sparse = K.get_sparse_matrix(t, 1e-10)
        m = m_sparse.toarray()

        s = [0.1, 0.2, 0.4]
        L = [1.0, 1.0, 1.2]

        expected = np.zeros((3, 3))
        for i, ti in enumerate(t):
            for j, tj in enumerate(t):
                expected[i, j] = math.exp(s[i]) * math.exp(s[j]) \
                    * np.sqrt(2 * math.exp(L[i]) * math.exp(L[j])
                              / (math.exp(L[i])**2 + math.exp(L[j])**2)) \
                    * math.exp(-abs(ti - tj)
                               / np.sqrt(math.exp(L[i])**2
                                         + math.exp(L[j])**2))

        assert np.allclose(m, expected)

    def test_sparse_matrix_cutoff(self):
        # Test getting a sparse matrix using cutoff method

        t = np.array([1.0, 2.0, 3.0])

        K = flexnoise.kernels.GPLaplacianKernel(
            np.array([1.0, 1.0, 1.2, 0.1, 0.2, 0.4]), t)

        K.sparse_method = 'cutoff'

        m_sparse = K.get_sparse_matrix(t, 1e-10)
        m = m_sparse.toarray()

        s = [0.1, 0.2, 0.4]
        L = [1.0, 1.0, 1.2]

        expected = np.zeros((3, 3))
        for i, ti in enumerate(t):
            for j, tj in enumerate(t):
                expected[i, j] = math.exp(s[i]) * math.exp(s[j]) \
                    * np.sqrt(2 * math.exp(L[i]) * math.exp(L[j])
                              / (math.exp(L[i])**2 + math.exp(L[j])**2)) \
                    * math.exp(-abs(ti - tj)
                               / np.sqrt(math.exp(L[i])**2
                                         + math.exp(L[j])**2))

        assert np.allclose(m, expected)

    def test_initialize_parameters(self):
        # Test the function which initializes parameters

        gp_times = np.linspace(0, 10, 10)
        K = flexnoise.kernels.GPLaplacianKernel([], gp_times)

        # Make independent residuals that increase in magnitude over time
        residuals = np.random.normal(0, 1, 1000) * np.linspace(1, 7, 1000)
        times = np.linspace(0, 10, 1000)

        K.initialize_parameters(times, residuals)

        # Check that sigma increases over time
        assert K.parameters[len(gp_times)] < K.parameters[-1]

        # Check that L is small, since the residuals were independent
        init_l1_corr = \
            math.exp(-abs(times[1] - times[0])
                     / np.mean(np.exp(K.parameters[:len(gp_times)])))
        assert init_l1_corr < 0.25
