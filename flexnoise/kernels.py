"""Covariance functions used in the flexible noise.
"""

import math
import numpy as np
import pints.residuals_diagnostics
import random
import scipy.interpolate
import scipy.signal
import scipy.sparse


class CovKernel:
    """Positive definite kernel, which can be used to generate covariances.
    """
    def __init__(self):
        super(CovKernel, self).__init__()

    def __call__(self, x, y):
        """The main kernel function k : R x R -> R.

        The result is conditional on the currently stored values of
        self.parameters. This function should handle vector inputs.

        Parameters
        ----------
        x : float or np.ndarray
            First argument
        y : float or np.ndarray
            Second argument

        Returns
        -------
        float or np.ndarray
            Kernel result for each pair of inputs.
        """
        raise NotImplementedError

    def get_matrix(self, t):
        """Get the Gramian matrix given a set of times.

        The result is conditional on the currently stored values of
        self.parameters.

        Parameters
        ----------
        t : np.ndarray
            The full set of times over which to calculate the covariance
            matrix.

        Returns
        -------
        np.ndarray
            2d array covariance matrix
        """
        raise NotImplementedError

    def num_parameters(self):
        """Get the number of parameters for this kernel.

        Returns
        -------
        int
            The number of scalar parameters for this kernel.
        """
        raise NotImplementedError

    def initialize_parameters(self):
        """Make an initial guess of the parameters based on given time series.

        This method is most useful for non-stationary kernels with parameters
        varying over time. This is an optional method, which helps inference
        for long time series by starting in a good location.

        Returns
        -------
        np.ndarray
            Guess of a good kernel parameter vector
        """
        raise NotImplementedError

    def get_sparse_matrix(self, t, threshold):
        """Get a sparse version of the covariance matrix.

        The result is conditional on the current values of self.parameters.

        Any diagonal whose maximum value is below the threshold will be set to
        all zero. The threshold must be set very low to avoid inaccuracy in the
        likelihood, around 1e-10 or lower depending on the problem.

        When possible, this function should generate the matrix in a highly
        scalable way rather than just generating the full matrix and then
        truncating.

        Parameters
        ----------
        t : np.ndarray
            The full set of times over which to calculate the covariance
            matrix.
        threshold : float
            Any diagonal all of whose values are below this number will be
            zeroed out. Must be a very small number.

        Returns
        -------
        scipy.sparse.csc.csc_matrix
            The covariance matrix in sparse csc form.
        """
        if self.sparse_method == 'diagonal':
            K = self._sparse_matrix_by_diagonal(t, threshold)
            return K
        elif self.sparse_method == 'cutoff':
            K = self._sparse_matrix_by_cutoff(t, threshold)
            return K

    def _sparse_matrix_by_diagonal(self, t, threshold):
        """Build the sparse covariance matrix one diagonal at a time.

        This method stops once the diagonals fall below the threshold. This is
        a fairly general method that should be applicable for many kernels.

        Parameters
        ----------
        t : np.ndarray
            The full set of times over which to calculate the covariance
            matrix.
        threshold : float
            Any diagonal all of whose values are below this number will be
            zeroed out. Must be a very small number.
        """
        lags = []
        diags = []
        lag = 0
        while True:
            if lag == 0:
                diag = self(t, t)
                # For numerical stability
                diag += 1e-14 * np.ones(len(diag))
            else:
                diag = self(t[lag:], t[:-lag])

            if len(diag) == 0:
                # Reached the end of the matrix
                break

            lags.append(lag)
            diags.append(diag)
            if lag != 0:
                lags.append(-lag)
                diags.append(diag)

            if max(diag) < threshold:
                break

            lag += 1

        n = len(t)
        cov_matrix = scipy.sparse.dia_matrix((n, n))
        for lag, diag in zip(lags, diags):
            cov_matrix.setdiag(diag, lag)
        return cov_matrix.tocsc()

    def _sparse_matrix_by_cutoff(self, t, threshold):
        """Alternative method for getting the sparse covariance matrix.

        Calculate the dense matrix, and then truncate values below the
        threshold to zero. Then, convert to sparse format.

        For covariance matrices with some large magnitude terms far from the
        diagonal, this method may be faster than the diagonal-by-diagonal
        method.
        """
        m = self.get_matrix(t)
        m[m < threshold] = 0
        return scipy.sparse.csc_matrix(m)


class LaplacianKernel(CovKernel):
    r"""The Laplacian positive definite kernel.

    The two parameters are a variance :math:`\sigma` and a length scale
    :math:`l`. The formula is

    .. math::
        k(x, y) = \sigma^2 e^{-|x - y| / L}

    This class uses the log-transformed sigma and L as its parameters.
    """
    def __init__(self, parameters):
        """
        Parameters
        ----------
        parameters : list
            [log(L), log(sigma)]
        """
        self.parameters = parameters
        self.sparse_method = 'diagonal'

    def __call__(self, x, y):
        log_L = self.parameters[0]
        log_sigma = self.parameters[1]
        sigma = math.exp(log_sigma)
        L = math.exp(log_L)
        return sigma**2 * np.exp(-np.abs(x-y) / L)

    def get_matrix(self, t):
        log_L = self.parameters[0]
        log_sigma = self.parameters[1]
        sigma = math.exp(log_sigma)
        L = math.exp(log_L)
        K = sigma**2 * np.exp(-np.abs(t - t[:, np.newaxis]) / L)

        # For numerical stability
        return K + 1e-14 * np.identity(len(t))

    def num_parameters(self):
        return 2


class GPLaplacianKernel(CovKernel):
    r"""Non-stationary Laplace kernel parameters varying over time.

    This is the non-stationary version of the Laplace kernel. The covariance
    between two inputs is given by

    .. math::
        C(x, y) = \sigma(x) \sigma(y) \sqrt{(2 L(x) L(y)) / (L(x)^2 + L(y)^2}
                \exp(-|x-y| / \sqrt{L(x)^2 + L(y)^2})

    The parameters for the kernel consist of vectors of :math:`L` and
    :math:`\sigma` along the length of the time series. This kernel is intended
    for use with a Gaussian process prior on the two parameter vectors.

    The time grid on which the parameters are defined is supplied when
    instantiating the class, and it may be sparser than the spacing of the
    actual data. Interpolation is used to populate the parameters on the
    denser time grid.
    """
    def __init__(self, parameters, times, transforms=None):
        """
        Parameters
        ----------
        parameters : np.ndarray
            A 1d array containing the log-transformed values of L and sigma.
        times : np.ndarray
            The times at which to infer GP values.
        transforms
            Optional matrix transformations to apply to the parameter vectors.
            If supplied, it should be a list of two square matrices which will
            be multiplied to the raw parameter vectors to obtain the log(L)
            and log(sigma) vectors.
        """
        self.parameters = parameters
        self.gp_times = times
        self.num_gp_times = len(times)

        if transforms is not None:
            self.transform = True
            self.transforms = transforms
        else:
            self.transform = False

        self.sparse_method = 'cutoff'

        self.cache = (None, None, None)

    def _parse_parameters(self, t):
        """Get the needed values of L and sigma, from the current parameters.

        This function handles exponentiation, transformation (if necessary),
        and interpolation of the non-stationary kernel parameters.

        The results are cached, so that later calls with the same parameters
        are faster.

        Parameters
        ----------
        t : float or np.ndarray
            The input locations at which to get the parameter values.

        Returns
        -------
        float or np.ndarray
            The values of L at the input locations
        float or np.ndarray
            The values of sigma at the input locations
        """
        if np.array_equal(self.parameters, self.cache[0]):
            # Get the results saved from a previous call
            f_L = self.cache[1]
            f_s = self.cache[2]

        else:
            # Get the logarithms of l, s on sparse grid
            log_L = self.parameters[:self.num_gp_times]
            log_s = self.parameters[self.num_gp_times:]

            if self.transform:
                log_L = self.transforms[0] @ log_L
                log_s = self.transforms[1] @ log_s

            L = np.exp(log_L)
            s = np.exp(log_s)

            # Interpolate to the full grid
            f_L = scipy.interpolate.interp1d(
                    self.gp_times, L, fill_value='extrapolate', kind='linear')

            f_s = scipy.interpolate.interp1d(
                    self.gp_times, s, fill_value='extrapolate', kind='linear')

        L = f_L(t)
        s = f_s(t)

        # Cache the results for faster performance if it is called again with
        # the same parameters
        self.cache = (self.parameters, f_L, f_s)

        return L, s

    def __call__(self, x, y):
        L, s = self._parse_parameters(np.array([x, y]))
        L_x, L_y = L
        s_x, s_y = s

        K = s_x * s_y * np.sqrt(2 * L_x * L_y / (L_x**2 + L_y**2)) \
            * np.exp(-np.abs(x-y) / np.sqrt(L_x**2 + L_y**2))
        return K

    def get_matrix(self, t):
        L, s = self._parse_parameters(t)

        K = np.outer(s, s) * \
            np.sqrt(2 * np.outer(L, L) / (L**2 + L[:, np.newaxis]**2)) * \
            np.exp(-np.abs(t - t[:, np.newaxis]) /
                   np.sqrt(L**2 + L[:, np.newaxis]**2))

        # For numerical stability
        return K + 1e-14 * np.identity(len(t))

    def num_parameters(self):
        return 2*len(self.gp_times)

    def initialize_parameters(self,
                              times,
                              residuals,
                              window_size_corr=None,
                              window_size_std=None,
                              truncate_small_values=True):
        """Rough guess of starting parameters from data.

        This function takes in a guess of the residuals, which could be
        estimated from an IID fit, and calculates values of L and sigma over
        time which match the data. These values can be used as a reasonable
        starting point for optimization or MCMC inference. They should not be
        considered equivalent to inferred values.

        Parameters
        ----------
        times : np.ndarray
            Time points for the residuals
        residuals : np.ndarray
            Guess of the residuals, observed data minus noise-free trajectory
        window_size_corr : int, optional
            Window size to use when calculating lag 1 autocorrelations. When
            not provided, set equal to 1/10 of the length of the time series.
        window_size_std : int, optional
            Window size to use when calculating standard deviations. When not
            provided, set equal to 1/10 of the length of the time series.
        truncate_small_values : bool, optional (False)
            Whether to push small values of empirical autocorrelation to zero
            before calculating initial parameters. If used, a random threshold
            is used.

        Returns
        -------
        np.ndarray
            Guess for the kernel parameter vector. They are also saved to
            self.parameters.
        """
        if window_size_corr is None:
            window_size_corr = math.ceil(len(times) / 10)
        if window_size_std is None:
            window_size_std = math.ceil(len(times) / 10)

        # Guess the values within each bin
        vars = []
        lag1_autocorrs = []
        for i, time in enumerate(times):
            L = len(times)
            # Get the window centered on this time point
            window_data_std = residuals[max(0, i - window_size_std):
                                        min(L, i + window_size_std)]
            vars.append(np.var(window_data_std))

            window_data_corr = residuals[max(0, i - window_size_corr):
                                         min(L, i + window_size_corr)]

            lag1_autocorrs.append(
                pints.residuals_diagnostics.acorr(window_data_corr, 1)[-1])

        # Smooth the binned estimates
        vars_filtered = scipy.signal.wiener(vars, window_size_std // 2 + 1)
        lag1_autocorrs_filtered = scipy.signal.wiener(
            lag1_autocorrs, window_size_corr // 2 + 1)

        # Interpolate to GP times
        f = scipy.interpolate.interp1d(times, vars_filtered)
        vars = f(self.gp_times)

        f = scipy.interpolate.interp1d(times, lag1_autocorrs_filtered)
        lag1_autocorrs = f(self.gp_times)

        # Calculate parameter values from the estimates
        ss_init = np.sqrt(vars)
        dt = times[1] - times[0]

        # Set negative or 0 autocorrs to a small number, 0.0001
        # This kernel parametrization cannot accept negative or zero
        # autocorrelations, but it can get close to zero.
        lag1_autocorrs[lag1_autocorrs < 0.0001] = 0.0001

        if truncate_small_values:
            # This step truncates low autocorrelation values to 0.0001. This
            # helps regions of small, probably spurious autocorrelation start
            # at good values of L. A small but random number is used to avoid
            # dependence on a particular threshold.
            lag1_autocorrs[lag1_autocorrs < random.uniform(0.25, 0.35)] = 1e-4

        ls_init = -dt / np.log(lag1_autocorrs)

        # Convert to logarithm
        ss_init = np.log(ss_init)
        ls_init = np.log(ls_init)

        if self.transform:
            ls_init = np.linalg.inv(self.transforms[0]) @ ls_init
            ss_init = np.linalg.inv(self.transforms[1]) @ ss_init

        params = np.concatenate((ls_init, ss_init))

        # Save to self, and return
        self.parameters = params

        return self.parameters
