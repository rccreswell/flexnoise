"""Generate figures used in the paper.
"""

import argparse
import flexnoise
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import scipy
from simulations import *


def plot_figure1(show_multivariate_normal=False,
                 output_dir='./'):
    """Make the introductory figure showing the two noise models.

    This is a cartoon figure which is only for illustrative purposes. It does
    not perform actual inference for any parameters or covariance.

    The figure will be saved in the provided directory with the filename
    'fig1.pdf'.

    Parameters
    ----------
    show_multivariate_normal : bool, optional (False)
        Whether or not to include a panel giving the equation of the
        multivariate normal distribution.
    output_dir : str, optional ('./')
        Path to the output directory in which to save the results.
    """
    random.seed(1212)
    np.random.seed(1212)

    # Set Latex and fonts
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = \
        r'\usepackage{{amsmath}}\renewcommand{\sfdefault}{phv}'

    # Make a synthetic data set
    t, v, x = generate_time_series(model='logistic',
                                   noise='graphic',
                                   n_times=50)

    # Get standard deviation over time for GP and block demonstrations
    def true_std(t):
        return 0.1 * v ** 1.2

    def block_std(t):
        return 3 * (t < 28) + 8 * ((t >= 28) & (t < 63)) + 15 * (t >= 63)

    # Build fake matrices which show the block vs smoothly varying models
    # These are modified from the true variance values in order to exaggerate
    # the difference between the two models.
    cov_gp = true_std(t)**2 * np.exp(-0.1 * np.abs(t - t[:, np.newaxis]))

    blocks = []
    for t_array in np.array_split(t, 3):
        block = block_std(t_array[10])**2 * \
                np.exp(-0.1 * np.abs(t_array - t_array[:, np.newaxis]))
        blocks.append(block)
    cov_block = scipy.linalg.block_diag(*blocks)

    fig = plt.figure(figsize=(9, 4.5))
    num_cols = 3 if show_multivariate_normal else 2
    grid_shape = (2, num_cols)

    if show_multivariate_normal:
        # Plot the data and model trajectory
        ax = fig.add_subplot(*grid_shape, 1)
        ax.plot(t, v, lw=2, label='Model', color='k')
        ax.scatter(t, x, marker='x', color='grey', alpha=1, label='Data')
        ax.set_xlabel('Time')
        ax.set_ylabel('y')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend()

        # Write the equation for the multivariate normal
        ax = fig.add_subplot(*grid_shape, 4)
        ax.text(0.46,
                0.55,
                r'$\begin{pmatrix} y_1 \\ \vdots \\ y_N '
                r'\end{pmatrix} \sim N\left( \begin{pmatrix} f_1 '
                r'\\ \vdots \\ f_N \end{pmatrix}, \Sigma\right)$',
                size=14)
        ax.set_xlim(0.45, 0.75)
        ax.set_ylim(0.45, 0.65)
        ax.set_axis_off()

    # Plot the data and noise fit for Gaussian process
    ax = fig.add_subplot(*grid_shape, 1 + int(show_multivariate_normal))
    ax.plot(t, v, lw=2, color='k', label='Model')
    ax.scatter(t,
               x,
               marker='o',
               s=10,
               color='grey',
               alpha=1,
               label='Data')
    ax.fill_between(t,
                    v - 2*true_std(t),
                    v + 2*true_std(t),
                    alpha=0.25,
                    color='grey',
                    label='Noise scale')
    ax.set_xlabel('Time')
    ax.set_ylabel('y')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Smoothly varying kernel\nparameters using Gaussian'
                 ' processes')
    if not show_multivariate_normal:
        ax.legend()

    # Plot the Gaussian process method covariance matrix
    ax = fig.add_subplot(*grid_shape, 3 + 2*int(show_multivariate_normal))
    ax.imshow(cov_gp, cmap='Greys')
    ax.set_ylabel(r'$\Sigma=  $', rotation=0, ha='right', size=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Covariance matrix')

    # Plot the data and noise fit for block method
    ax = fig.add_subplot(*grid_shape, 2 + int(show_multivariate_normal))
    ax.plot(t, v, lw=2, color='k', label='Model')
    ax.scatter(t,
               x,
               marker='o',
               s=10,
               color='grey',
               alpha=1,
               label='Data')
    t_dense = np.linspace(0, 100, 10000)
    v_dense = scipy.interpolate.interp1d(t, v)(t_dense)
    ax.fill_between(t_dense,
                    v_dense - 2*block_std(t_dense),
                    v_dense + 2*block_std(t_dense),
                    alpha=0.25,
                    color='grey',
                    label='Noise scale')
    ax.set_xlabel('Time')
    ax.set_ylabel('y')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Nonparametric ``blocked''\nkernel parameters")
    if not show_multivariate_normal:
        ax.legend()

    # Plot the block method covariance matrix
    ax = fig.add_subplot(*grid_shape, 4 + 2*int(show_multivariate_normal))
    ax.imshow(cov_block, cmap='Greys')
    ax.set_ylabel(r'$\Sigma=  $', rotation=0, ha='right', size=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Covariance matrix')

    # Add labels and adjust sizing
    text_transform = fig.transFigure
    if show_multivariate_normal:
        plt.text(0.1, 0.925, 'A.', fontsize=18, transform=text_transform)
        plt.text(0.37, 0.925, 'B.', fontsize=18, transform=text_transform)
        plt.text(0.66, 0.925, 'C.', fontsize=18, transform=text_transform)
        fig.set_tight_layout(True)
        plt.subplots_adjust(hspace=0.35)
        plt.subplots_adjust(wspace=0.25)

    else:
        plt.text(0.1, 0.925, 'A.', fontsize=18, transform=text_transform)
        plt.text(0.52, 0.925, 'B.', fontsize=18, transform=text_transform)

    fname = os.path.join(output_dir, 'fig1.pdf')
    plt.savefig(fname)


def plot_figureS3(output_dir='./'):
    """Make a figure showing some slices of the block prior.

    This figure is to show some properties of the prior over partitions used
    for learning block covariance matrices. It plots some probability mass
    functions over the number of blocks. See [1]_ for further details.

    The figure will be saved in the provided directory with the filename
    'figS3.pdf'.

    Parameters
    ----------
    output_dir : str, optional ('./')
        Path to the output directory in which to save the results.

    References
    ----------
    .. [1] Martinez, Asael Fabian, and Ramses H. Mena. "On a nonparametric
           change point detection model in Markovian regimes." Bayesian
           Analysis 9.4 (2014): 823-858.
    """
    # Set Latex and fonts
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = \
        r'\usepackage{{amsmath}}\renewcommand{\sfdefault}{phv}'

    n = 60  # Number of timepoints, also the maximum number of blocks
    k = np.arange(1, n+1)  # All possible values for number of blocks

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(2, 1, 1)

    # Fix sigma=0.5, show the prior for 3 different values of phi
    phis = [-0.25, 0.25, 5]
    styles = ['-', '--', ':']
    for phi, ls in zip(phis, styles):
        ax.plot(k,
                flexnoise.prior_K(k, n, 0.5, phi),
                ls=ls,
                color='grey',
                label=r'$\phi = {}$'.format(phi))

    ax.legend()
    ax.set_ylabel('Probability')
    ax.set_xlabel('Blocks')
    ax.set_xticks([1, 20, 40, 60])

    ax = fig.add_subplot(2, 1, 2)

    # Fix mean blocks = 40, and show different values of sigma. For each value
    # of sigma, theta is adjusted to get that prior mean
    sigmas = [0.3, 0.5, 0.7]
    for sigma, ls in zip(sigmas, styles):
        phi = flexnoise.find_prior_mean_theta(n, sigma, num_blocks=20)
        ax.plot(k,
                flexnoise.prior_K(k, n, sigma, phi),
                ls=ls,
                color='grey',
                label=r'$\sigma = {}$'.format(sigma))

    ax.legend()
    ax.set_ylabel('Probability')
    ax.set_xlabel('Blocks')
    ax.set_xticks([1, 20, 40, 60])

    fig.set_tight_layout(True)

    fname = os.path.join(output_dir, 'figS3.pdf')
    plt.savefig(fname)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir',
                        help='output directory',
                        default='results',
                        nargs='?')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    plot_figure1(output_dir=args.output_dir)
    plot_figureS3(output_dir=args.output_dir)


if __name__ == '__main__':
    main()
