"""Bayesian inference for model parameters.
"""

import argparse
import flexnoise
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pints
import pints.toy
import random
from simulations import *
import warnings


def run_figure2(num_mcmc_samples=20000,
                num_mcmc_chains=3,
                num_runs=8,
                output_dir='./'):
    """Run the Gaussian process on multiplicative data.

    This function runs the simulations and saves the results to pickle.
    """
    random.seed(123)
    np.random.seed(123)

    all_fits = []
    iid_runs = []
    sigmas = []
    mult_runs = []
    gp_runs = []
    for run in range(num_runs):
        # Make a synthetic time series
        times, values, data = generate_time_series(model='logistic',
                                                   noise='multiplicative',
                                                   n_times=251)

        # Make Pints model and problem
        model = pints.toy.LogisticModel()
        problem = pints.SingleOutputProblem(model, times, data)

        # Initial conditions for model parameters
        model_starting_point = [0.08, 50]

        # Run MCMC for IID posterior
        likelihood = pints.GaussianLogLikelihood
        x0 = model_starting_point + [2]
        posterior_iid = run_pints(
                            problem, likelihood, x0, num_mcmc_samples)
        iid_runs.append(posterior_iid)

        # Save standard deviations from IID runs
        sigma = np.median(posterior_iid[:, 2])
        sigmas.append(sigma)

        # Run MCMC for multiplicative noise posterior
        likelihood = pints.MultiplicativeGaussianLogLikelihood
        x0 = model_starting_point + [0.5, 0.5]
        posterior_mult = run_pints(
                            problem, likelihood, x0, num_mcmc_samples)
        mult_runs.append(posterior_mult)

        # Infer the nonstationary kernel fit
        # Run an optimization assumming IID
        log_prior = pints.UniformLogPrior([0] * 3, [1e6] * 3)
        log_likelihood = pints.GaussianLogLikelihood(problem)
        log_posterior = pints.LogPosterior(log_likelihood, log_prior)
        opt = pints.OptimisationController(
            log_posterior, model_starting_point + [2])
        xbest, fbest = opt.run()

        # Run the GP fit, using the best fit for initialization
        gp_times = times[::10]
        kernel = flexnoise.kernels.GPLaplacianKernel
        gnp = flexnoise.GPNoiseProcess(
            problem,
            kernel,
            xbest[:2],
            gp_times
        )
        gnp.set_gp_hyperparameters(mu=0.0, alpha=1.0, beta_num_points=200)
        x = gnp.run_optimize(num_restarts=100, parallel=True, maxiter=150)
        all_fits.append(x)

        # Run MCMC for multivariate normal noise
        kernel = flexnoise.kernels.GPLaplacianKernel(None, gp_times)
        kernel.parameters = x[2:]
        cov = kernel.get_matrix(times)
        likelihood = flexnoise.CovarianceLogLikelihood
        x0 = model_starting_point
        posterior_gp = run_pints(
            problem,
            likelihood,
            x0,
            num_mcmc_samples,
            likelihood_args=[cov]
        )
        gp_runs.append(posterior_gp)

    # Save all results to pickle
    results = [iid_runs, mult_runs, all_fits, gp_runs, times, data, values,
               model, problem, kernel, sigmas]

    fname = os.path.join(output_dir, 'fig2_data.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(results, f)


def plot_figure2(output_dir='./'):
    """Plot the results for figure 2.

    This function expects that the results have already been run and saved in
    the given output_dir as fig2_data.pkl.
    """
    fname = os.path.join(output_dir, 'fig2_data.pkl')
    with open(fname, 'rb') as f:
        iid_runs, mult_runs, all_fits, gp_runs, times, data, values, model, \
            problem, kernel, sigmas = pickle.load(f)

    # Set Latex and fonts
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = \
        r'\usepackage{{amsmath}}\renewcommand{\sfdefault}{phv}'

    # Set ground truth noise over time
    def true_std(t):
        return 0.0075 * values ** 2

    def true_autocorr(t):
        return 0 * t

    # Plot the fit to the time series
    fname = os.path.join(output_dir, 'fig2a.pdf')
    flexnoise.plot.plot_nonstat_fit(
        problem,
        kernel=kernel,
        samples=all_fits,
        style='lines',
        fname=fname,
        true_std=true_std,
        true_autocorr=true_autocorr,
        hlines=sigmas
    )

    # Plot all of the model parameter posteriors
    fname = os.path.join(output_dir, 'fig2b.pdf')
    flexnoise.plot.plot_grouped_parameter_posteriors(
        mult_runs,
        gp_runs,
        iid_runs,
        true_model_parameters=[0.08, 50],
        method_names=['Multiplicative (Correct)', 'GP Laplacian', 'IID'],
        parameter_names=['r', 'K'],
        fname=fname
    )


def run_figure3(num_mcmc_samples=20000,
                num_mcmc_chains=3,
                num_runs=1,
                output_dir='./'):
    """Run the block noise process.

    This function runs the simulations and saves the results to pickle.
    """
    random.seed(1234)
    np.random.seed(1234)

    iid_runs = []
    correct_infer_runs = []
    block_runs_theta = []
    block_runs_cov = []
    for run in range(num_runs):
        # Make a synthetic time series
        times, values, data = generate_time_series(model='logistic',
                                                   noise='blocks',
                                                   n_times=500)

        # Make Pints model and problem
        model = pints.toy.LogisticModel()
        problem = pints.SingleOutputProblem(model, times, data)

        # Initial conditions for model parameters
        model_starting_point = [0.08, 50]

        # Run MCMC for IID posterior
        likelihood = pints.GaussianLogLikelihood
        x0 = model_starting_point + [2]
        posterior_iid = run_pints(
                            problem, likelihood, x0, num_mcmc_samples)
        iid_runs.append(posterior_iid)

        # Run with fixed correct blocks
        likelihood = flexnoise.KnownBlocksCovarianceLogLikelihood
        kernel = flexnoise.kernels.LaplacianKernel(None)
        blocks = []
        for i in range(5):
            blocks.append(list(range(100*i, 100*i+100)))

        x0 = model_starting_point.copy()
        for i in range(len(blocks)):
            x0 += [-1.0, 0.0]
        log_prior = pints.UniformLogPrior([0, 0] + [-1e6]*10, [1e6]*12)
        posterior_known_blocks_infer = run_pints(
                            problem, likelihood, x0, num_mcmc_samples,
                            likelihood_args=[kernel, blocks, True],
                            log_prior=log_prior)

        posterior_known_blocks_infer = posterior_known_blocks_infer[:, :2]
        correct_infer_runs.append(posterior_known_blocks_infer)

        # Run block noise process
        num_mcmc_samples = 200
        model_prior = pints.UniformLogPrior([0] * 2, [1e6] * 2)
        kernel = flexnoise.kernels.LaplacianKernel
        bnp = flexnoise.BlockNoiseProcess(
            problem, kernel, np.array(model_starting_point), model_prior)
        theta_chains = []
        cov_chains = []
        for _ in range(num_mcmc_chains):
            theta, cov = bnp.run_mcmc(num_mcmc_samples)
            theta_chains.append(np.array(theta))
            cov_chains.append(np.array(cov))
        rs = pints.rhat(np.array(theta_chains))
        if max(rs) > 1.05:
            warnings.warn('MCMC chains failed to converge')
        block_runs_theta.append(theta)
        block_runs_cov.append(cov)

    # Save all results to pickle
    results = [iid_runs, correct_infer_runs, block_runs_theta,
               block_runs_cov, times, data, values, model, problem]

    fname = os.path.join(output_dir, 'fig3_data.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(results, f)


def plot_figure3(output_dir='./'):
    """Plot the results for figure 3.

    This function expects that the results have already been run and saved in
    the given output_dir as fig3_data.pkl.
    """
    fname = os.path.join(output_dir, 'fig3_data.pkl')
    with open(fname, 'rb') as f:
        iid_runs, correct_infer_runs, block_runs_theta, block_runs_cov, \
            times, data, values, model, problem = pickle.load(f)

    # Set Latex and fonts
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = \
        r'\usepackage{{amsmath}}\renewcommand{\sfdefault}{phv}'

    # Set ground truth noise over time
    def true_std(t):
        return 3 * ((t < 60) | (t > 80)) + 30 * ((t > 60) & (t < 80))

    def true_autocorr(t):
        return 0 * ((t < 20) | (t > 40)) + 0.85 * ((t > 20) & (t < 40))

    # Plot the fit to the time series
    for run, (theta_chain, cov_chain) in \
            enumerate(zip(block_runs_theta, block_runs_cov)):
        fname = os.path.join(output_dir, 'fig3a_{}.pdf'.format(run))
        flexnoise.plot.plot_nonstat_fit(
            problem,
            samples=np.array(theta_chain).T,
            cov_samples=cov_chain,
            style='dist',
            fname=fname,
            true_std=true_std,
            true_autocorr=true_autocorr,
        )

    # Plot all of the model parameter posteriors
    fname = os.path.join(output_dir, 'fig3b.pdf')
    flexnoise.plot.plot_grouped_parameter_posteriors(
        correct_infer_runs,
        block_runs_theta,
        iid_runs,
        true_model_parameters=[0.08, 50],
        method_names=['Correct', 'Block', 'IID'],
        parameter_names=['r', 'K'],
        fname=fname
    )


def run_figureS2(num_runs=3,
                 output_dir='./'):
    """Run the Gaussian process on block noise data.

    This function runs the simulations and saves the results to pickle.
    """
    random.seed(12345)
    np.random.seed(12345)

    all_fits = []
    iid_runs = []
    sigmas = []
    mult_runs = []
    gp_runs = []
    for run in range(num_runs):
        # Make a synthetic time series
        times, values, data = generate_time_series(model='logistic',
                                                   noise='blocks',
                                                   n_times=625)

        # Make Pints model and problem
        model = pints.toy.LogisticModel()
        problem = pints.SingleOutputProblem(model, times, data)

        # Initial conditions for model parameters
        model_starting_point = [0.08, 50]

        # Infer the nonstationary kernel fit
        # Run an optimization assumming IID
        log_prior = pints.UniformLogPrior([0] * 3, [1e6] * 3)
        log_likelihood = pints.GaussianLogLikelihood(problem)
        log_posterior = pints.LogPosterior(log_likelihood, log_prior)
        opt = pints.OptimisationController(
            log_posterior, model_starting_point + [2])
        xbest, fbest = opt.run()

        # Run the GP fit, using the best fit for initialization
        gp_times = times[::25]
        kernel = flexnoise.kernels.GPLaplacianKernel
        gnp = flexnoise.GPNoiseProcess(
            problem,
            kernel,
            xbest[:2],
            gp_times
        )
        gnp.set_gp_hyperparameters(mu=0.0, alpha=1.0, beta_num_points=200)
        x = gnp.run_optimize(num_restarts=100, parallel=True, maxiter=150)
        all_fits.append(x)

    # Save all results to pickle
    kernel = kernel(None, gp_times)
    results = [all_fits, times, data, values, model, problem, kernel]

    fname = os.path.join(output_dir, 'figS2_data.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(results, f)


def plot_figureS2(output_dir='./'):
    """Plot the results for figure 2.

    This function expects that the results have already been run and saved in
    the given output_dir as fig2_data.pkl.
    """
    fname = os.path.join(output_dir, 'figS2_data.pkl')
    with open(fname, 'rb') as f:
        all_fits, times, data, values, model, problem, kernel = pickle.load(f)

    # Set Latex and fonts
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = \
        r'\usepackage{{amsmath}}\renewcommand{\sfdefault}{phv}'

    # Set ground truth noise over time
    def true_std(t):
        return 3 * ((t < 60) | (t > 80)) + 30 * ((t > 60) & (t < 80))

    def true_autocorr(t):
        return 0 * ((t < 20) | (t > 40)) + 0.85 * ((t > 20) & (t < 40))

    # Plot the fit to the time series
    fname = os.path.join(output_dir, 'figS2.pdf')
    flexnoise.plot.plot_nonstat_fit(
        problem,
        kernel=kernel,
        samples=all_fits,
        style='lines',
        fname=fname,
        true_std=true_std,
        true_autocorr=true_autocorr
    )


def make_figureS1(num_runs=10,
                  num_mcmc_samples=20000,
                  num_mcmc_chains=3,
                  output_dir='./'):
    """Compares parameter posteriors with IID, AR(1), and Laplace kernel.

    This figure contains two panels, each of which is saved in its own PDF
    file. The first panel uses the logistic growth model and plots one
    trajectory with AR(1) noise and one trajectory with IID noise to show the
    difference. The second panel shows the posterior distributions for the two
    logistic growth model parameters on data with AR(1) noise under three
    different specifications of the noise model: the correct AR(1), the
    incorrect IID, and the more flexible stationary Laplace kernel.

    Parameters
    ----------
    num_runs : int
        The number of replicates to run.
    num_mcmc_samples : int
        The number of MCMC samples in each chain.
    num_mcmc_chains : int
        The number of separate MCMC chains to use for each replicate to check
        convergence.
    output_dir : str
        Path to the output directory in which to save the results.
    """
    random.seed(123)
    np.random.seed(123)

    # Create the output directory if it does not exist
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Set Latex and fonts
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = \
        r'\usepackage{{amsmath}}\renewcommand{\sfdefault}{phv}'

    # Make an IID time series for comparison in the figure
    times_iid, values_iid, data_iid = generate_time_series(model='logistic',
                                                           noise='IID')

    iid_runs = []
    ar1_runs = []
    lap_runs = []
    for run in range(num_runs):
        # Make synthetic data
        times, values, data = generate_time_series(model='logistic',
                                                   noise='AR1')

        if run == 0:
            # Make a plot of IID vs AR(1) data
            fig = plt.figure(figsize=(6, 3))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(times, values, label='Noise-free trajectory', color='k')
            ax.scatter(
                times, data, label='Data', color='grey', s=6.0, alpha=0.5)
            ax.set_title('AR(1) noise')
            ax.set_xlabel('Time')
            ax.set_ylabel('y')
            ax.legend()

            fig.set_tight_layout(True)
            fname = os.path.join(output_dir, 'figS1a.pdf')
            plt.savefig(fname)

        # Make Pints model and problem
        model = pints.toy.LogisticModel()
        problem = pints.SingleOutputProblem(model, times, data)

        # Initial conditions for model parameters
        model_starting_point = [0.08, 50]

        # Noise 1: assume IID Gaussian
        likelihood = pints.GaussianLogLikelihood
        x0 = model_starting_point + [2]
        posterior_iid = run_pints(problem, likelihood, x0, num_mcmc_samples)

        # Noise 2: assume ar1
        likelihood = pints.AR1LogLikelihood
        x0 = model_starting_point + [0.5, 0.5]
        prior = pints.UniformLogPrior([0, 0, 0, 0], [1e6, 1e6, 0.99, 1e6])
        posterior_ar1 = run_pints(
                            problem,
                            likelihood,
                            x0,
                            num_mcmc_samples,
                            log_prior=prior)

        # Noise 3: use kernel noise
        likelihood = flexnoise.KernelCovarianceLogLikelihood
        kernel = flexnoise.kernels.LaplacianKernel([-2, -2])
        x0 = model_starting_point + [-2, -2]
        num_params = len(x0)
        prior = pints.UniformLogPrior([-1e6] * num_params, [1e6] * num_params)
        chain_kernel = run_pints(
                        problem,
                        likelihood,
                        x0,
                        num_mcmc_samples,
                        log_prior=prior,
                        likelihood_args=[kernel])

        iid_runs.append(posterior_iid)
        ar1_runs.append(posterior_ar1)
        lap_runs.append(chain_kernel)

    # Plot all of the model parameter posteriors
    fname = os.path.join(output_dir, 'figS1b.pdf')
    flexnoise.plot.plot_grouped_parameter_posteriors(
                ar1_runs,
                lap_runs,
                iid_runs,
                true_model_parameters=[0.08, 50],
                method_names=['AR(1) (Correct)', 'Laplacian kernel', 'IID'],
                parameter_names=['r', 'K'],
                fname=fname)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir',
                        help='output directory',
                        default='results',
                        nargs='?')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # run_figure2(output_dir=args.output_dir)
    plot_figure2(output_dir=args.output_dir)
    # run_figure3(output_dir=args.output_dir)
    plot_figure3(output_dir=args.output_dir)
    # make_figureS1(output_dir=args.output_dir)
    # run_figureS2(output_dir=args.output_dir)
    plot_figureS2(output_dir=args.output_dir)


if __name__ == '__main__':
    main()
