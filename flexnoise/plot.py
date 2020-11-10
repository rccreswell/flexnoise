"""Functions for visualizing time series with flexible noise processes.
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse


def plot_grouped_parameter_posteriors(
        *chains,
        true_model_parameters=None,
        colors=['white', 'lightblue', 'grey', 'darkblue'],
        method_names=None,
        parameter_names=None,
        fname='posterior.pdf'):
    """Plot the posterior distributions from different methods.

    This figure creates one panel for each model parameter, and groups the
    posteriors for each method by replicate along the x axis.

    Parameters
    ----------
    *chains : np.ndarry
        MCMC chains containing the samples. For each method, the MCMC chains in
        one numpy array should be provided.
    true_model_parameters : list of float, optional (None)
        The ground truth values of the model parameters, to be drawn on the
        plot.
    colors : list of str, optional
        Colors to use for labelling the posterior from each method. Must have
        length equal to the number of chains.
    method_names : list of str, optional (None)
        List giving the name of each method.
    parameter_names : list of str, optional (None)
        List giving the name of each model parameter.
    fname : str, optional ('posterior.pdf')
        Filename to save the figure. If None, the figure is not saved but
        returned.
    """
    num_model_parameters = len(true_model_parameters)

    fig = plt.figure(figsize=(6, 4.5))

    for i in range(num_model_parameters):
        # Make one panel for this model parameter
        ax = fig.add_subplot(num_model_parameters, 1, i+1)
        legend_boxes = []

        for j in range(len(chains)):
            # Plot the posterior from this method
            all_samplers = np.array(chains[j])
            samples = all_samplers[:, :, i]
            num_runs = all_samplers.shape[0]
            positions = (np.arange(num_runs) * (len(chains) + 2)) + 1 + j

            # Get the middle chain to use as the location of the run label
            if j == len(chains) // 2:
                tick_positions = positions

            # Settings for boxplot
            medianprops = dict(linewidth=0)

            # Plot all runs from this method and model parameter
            boxes = ax.boxplot(samples.T,
                               positions=positions,
                               sym='',
                               whis=[2.5, 97.5],
                               medianprops=medianprops,
                               patch_artist=True)

            for patch in boxes['boxes']:
                patch.set_facecolor(colors[j])

            # Add a box of the appropriate color to the legend
            legend_box = mpatches.Patch(facecolor=colors[j],
                                        label=method_names[j],
                                        edgecolor='black',
                                        linewidth=1,
                                        linestyle='-')
            legend_boxes.append(legend_box)

        if i == 0:
            # Add a legend above the plot
            ax.legend(handles=legend_boxes,
                      loc='upper center',
                      bbox_to_anchor=(0.5, 1.2),
                      ncol=3)

        ax.axhline(true_model_parameters[i], ls='--', color='k')
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([])
        ax.set_ylabel(parameter_names[i])

    ax.set_xlabel('Replicate')
    ax.set_xticklabels(np.arange(num_runs) + 1)
    fig.set_tight_layout(True)

    if fname is not None:
        plt.savefig(fname)

    return fig


def plot_nonstat_fit(problem,
                     kernel=None,
                     samples=None,
                     cov_samples=None,
                     style='dist',
                     model_parameters=None,
                     fname='noise_fit.pdf',
                     true_std=None,
                     true_autocorr=None,
                     hlines=None,
                     trajcolor=None):
    """Plot a nonstationary covariance fit.

    This figure contains three panels. The top panel shows the data and the
    model fits. The middle panel shows the variance over time (diagonal of
    covariance matrix). The bottom panel shows the lag 1 autocorrelation over
    time.

    This function accepts either a single best-fit optimization result or a set
    of posterior samples from a MCMC chain.

    Parameters
    ----------
    problem : pints.SingleOutputProblem
        The time series problem in Pints format
    kernel : flexnoise.CovKernel, optional (None)
        Non stationary covariance function
    samples : np.ndarray, optional (None)
        Values of the model parameters and kernel parameters. For n_m model
        parameters and n_k total kernel parameters, this input may take any of
        the following shapes:
        (n_m + n_k, ) - 1 sample of both model and kernel parameters
        (n_m + n_k, N) - N MCMC samples or optimization results of both model
                        and kernel parameters
        (n_k, ) - One sample of kernel parameters. Fixed model parameters must
                    be supplied in the model_parameters.
        (n_k, N) - N MCMC samples or optimzation results of kernel parameters.
                    Fixed model parameters must be supplied in the
                    model_parameters.
    cov_samples : list, optional (None)
        List of samples of the covariance matrix. Each item in the list can be
        either a 2d numpy array, or a scipy sparse csc matrix. If this is
        supplied, samples should consist only of values for the model
        parameters.
    style : {'dist', 'lines'}, optional ('dist')
        Whether to plot the samples as a posterior distribution or as
        individual lines.
    model_parameters : np.ndarray
        Fixed values of the model parameters to use. This is required when
        samples only contains kernel parameter values.
    fname : str, optional ('noise_fit.pdf')
        Filename to save the figure. If None, the figure is not saved but
        returned.
    true_std : function, optional (None)
        True standard deviation as a function of time
    true_autocorr : function, optional (None)
        True lag 1 autocorrelation as a function of time
    hlines : list of float, optional (None)
        Values to plot as dotted lines on the horizontal axis in the standard
        deviation plot
    trajcolor : str, optional (None)
        Color to use for model fits
    """
    times = problem.times()
    data = problem.values()
    if cov_samples is None:
        if type(samples) is list:
            samples = np.array(samples)

        try:
            n_params = samples.shape[1]

        except IndexError:
            n_params = samples.shape[0]
            samples = samples[np.newaxis, :]

        # Check samples shape and get the parameters of the Pints forward model
        if n_params == problem.n_parameters() + kernel.num_parameters():
            model_params = samples[:, :problem.n_parameters()]
            kernel_params = samples[:, problem.n_parameters():]
        elif n_params == kernel.num_parameters():
            model_params = [model_parameters]
            kernel_params = samples
            if model_params is None:
                raise ValueError('No model parameters found')
        else:
            raise ValueError('samples has the wrong shape')

    elif cov_samples is not None:
        model_params = samples.T

    # Get trajectories of model fits
    model_fits = []
    for model_param in model_params:
        model_fit = problem.evaluate(model_param)
        model_fits.append(model_fit)

    if style == 'dist':
        # Get the percentiles and median of all model fits
        model_fit_mid = np.median(model_fits, axis=0)
        model_fit_95 = np.percentile(model_fits, 95, axis=0)
        model_fit_5 = np.percentile(model_fits, 5, axis=0)

    # Get trajectories of noise fits
    std_fits = []
    autocorr_fits = []
    if cov_samples is None:
        for kernel_param in kernel_params:
            kernel.parameters = kernel_param.copy()
            cov = kernel.get_matrix(times)
            var = np.diag(cov)
            cov_lag1 = np.diag(cov, 1)
            autocorr_lag1 = cov_lag1 / (np.sqrt(var[1:]) * np.sqrt(var[:-1]))
            std_fits.append(np.sqrt(var))
            autocorr_fits.append(autocorr_lag1)

    else:
        for cov in cov_samples:
            if type(cov) is scipy.sparse.csc_matrix:
                cov = cov.toarray()
            var = np.diag(cov)
            cov_lag1 = np.diag(cov, 1)
            autocorr_lag1 = cov_lag1 / (np.sqrt(var[1:]) * np.sqrt(var[:-1]))
            std_fits.append(np.sqrt(var))
            autocorr_fits.append(autocorr_lag1)

    if style == 'dist':
        # Get the percentiles and median of all noise fits
        std_mid = np.median(std_fits, axis=0)
        std_95 = np.percentile(std_fits, 95, axis=0)
        std_5 = np.percentile(std_fits, 5, axis=0)
        autocorr_mid = np.median(autocorr_fits, axis=0)
        autocorr_95 = np.percentile(autocorr_fits, 95, axis=0)
        autocorr_5 = np.percentile(autocorr_fits, 5, axis=0)

    # Draw the figure
    fig = plt.figure(figsize=(6, 5.5))
    ax = fig.add_subplot(3, 1, 1)

    if trajcolor is None:
        trajcolor = 'k'

    dataplot = ax.scatter(times,
                          data,
                          s=6.0,
                          label='Data',
                          color='grey',
                          zorder=-100,
                          alpha=0.5)
    dataplot = [dataplot]

    fill = None
    if style == 'dist':
        modelplot = ax.plot(times,
                            model_fit_mid,
                            color='k',
                            label='Model fit')
        ax.plot(times, model_fit_5, color='k')
        ax.plot(times, model_fit_95, color='k')
        ax.fill_between(times,
                        model_fit_5,
                        model_fit_95,
                        color='k',
                        alpha=0.25)

        ax.fill_between(times,
                        model_fit - 2 * std_mid,
                        model_fit + 2 * std_mid,
                        color='grey',
                        alpha=0.35)
        fill = 1

    elif style == 'lines':
        for i, model_fit in enumerate(model_fits):
            label = 'Model fit' if i == 0 else None
            if i == 0:
                modelplot = ax.plot(
                    times, model_fit, color=trajcolor, label=label, alpha=0.65)
            else:
                ax.plot(
                    times, model_fit, color=trajcolor, label=label, alpha=0.65)

            if i == 0:
                ax.fill_between(times,
                                model_fit - 2 * std_fits[i],
                                model_fit + 2 * std_fits[i],
                                color='grey',
                                alpha=0.35,
                                zorder=-100)
        fill = 1

    ax.set_ylabel('y')

    ax = fig.add_subplot(3, 1, 2)

    if true_std is not None:
        truthplot = ax.plot(times,
                            true_std(times),
                            ls='--',
                            color='black',
                            label='Ground truth')

    if style == 'dist':
        ax.plot(times, std_mid, label='Standard deviation', color='k')
        ax.fill_between(times, std_5, std_95, color='grey', alpha=0.35)
        ax.set_ylim(-0.05, 1.05 * max(std_95))

    elif style == 'lines':
        for i, std_fit in enumerate(std_fits):
            label = 'Standard deviation' if i == 0 else None
            ax.plot(times,
                    std_fit,
                    color=trajcolor,
                    label=label,
                    lw=1.1,
                    alpha=0.65)
            ax.set_ylim(-0.05, 1.05 * np.max(std_fits))

    if hlines is not None:
        for i, h in enumerate(hlines):
            iidplot = ax.axhline(h,
                                 color='k',
                                 ls=':',
                                 lw=0.75,
                                 label='Model fit (IID)',
                                 alpha=0.65)

    ax.set_ylabel('Std. deviation')

    ax = fig.add_subplot(3, 1, 3)

    if style == 'dist':
        ax.plot(times[:-1],
                autocorr_mid,
                label='Lag 1 autocorrelation',
                color='k')
        ax.fill_between(times[:-1],
                        autocorr_5,
                        autocorr_95,
                        color='grey',
                        alpha=0.35)

    elif style == 'lines':
        for i, autocorr_fit in enumerate(autocorr_fits):
            label = 'Lag 1 autocorrelation' if i == 0 else None
            ax.plot(times[:-1],
                    autocorr_fit,
                    color=trajcolor,
                    label=label,
                    lw=1.1,
                    alpha=0.65)

    if true_autocorr is not None:
        ax.plot(times,
                true_autocorr(times),
                ls='--',
                color='black',
                label='Ground truth',
                zorder=-10)

    try:
        objects = dataplot + modelplot + truthplot
    except UnboundLocalError:
        objects = dataplot + modelplot
    if hlines is not None:
        objects += [iidplot]
    if fill is not None:
        # Add a box of the appropriate color to the legend
        legend_box = mpatches.Patch(facecolor='grey',
                                    label='Uncertainty',
                                    edgecolor='black',
                                    linewidth=1,
                                    linestyle='-',
                                    alpha=0.35)
        objects.append(legend_box)
    labels = [x.get_label() for x in objects]

    ax.legend(objects,
              labels,
              loc='upper center',
              bbox_to_anchor=(0.5, 3.85),
              ncol=3)
    ax.set_xlabel('Time')
    ax.set_ylabel('Lag 1\nAutocorrelation')
    ax.set_ylim(-0.05, 1)

    if fname is not None:
        plt.savefig(fname)

    return fig
