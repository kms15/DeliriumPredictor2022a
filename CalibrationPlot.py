#!/usr/bin/python3

import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.utils import resample
import time
import os.path
import argparse
import pickle
import warnings

startup_time = time.time()

def calibration_curve(bin_boundaries, ys, y_hats):
    n_bins = len(bin_boundaries) - 1
    bins = np.digitize(y_hats, bin_boundaries) - 1
    bin_pos = np.bincount(bins, ys, minlength=n_bins)
    bin_est_pos = np.bincount(bins, y_hats, minlength=n_bins)
    bin_counts = np.bincount(bins, minlength=n_bins).astype(float)

    ece = np.sum(np.abs(bin_pos - bin_est_pos)) / len(ys)

    # avoid warning when dividing by a bin of size 0
    divisor_bin_counts = bin_counts.copy()
    divisor_bin_counts[bin_counts == 0] = np.nan # empty bin means no valid data
    frac_pos = bin_pos / divisor_bin_counts

    bin_calibration_error = np.abs(bin_pos - bin_est_pos)/divisor_bin_counts
    mce = np.max(bin_calibration_error[bin_counts != 0])

    return frac_pos, ece, mce


def bootstrapped_calibration_plot(ax, ys, y_hats,
        xlabel="Predicted Probability",
        ylabel=("Observed Probability","Number of events in bin"), legend=True,
        num_bootstraps=1000, n_bins=10):

    alpha = 0.05 # 1 - (confidence interval = 95%)

    bin_boundaries = np.arange(n_bins + 1) / n_bins
    bin_boundaries[-1] = 1.0 + np.spacing(1.0) # make sure the final bin includes 1.0
    bin_centers = (np.arange(n_bins) + 0.5) / n_bins

    # generate the data for the main calibration curve
    fractions_of_positives, ece, mce = calibration_curve(bin_boundaries, ys, y_hats)

    # generate the bootstrap samples
    bootstrap_samples = [resample(ys, y_hats) for i in range(num_bootstraps)]

    bootstrap_fractions_of_positives = []
    bootstrap_ece = []
    bootstrap_mce = []
    for ys_,y_hats_ in bootstrap_samples:
        fractions_of_positives_, ece_, mce_ = calibration_curve(bin_boundaries, ys_, y_hats_)
        bootstrap_fractions_of_positives.append(fractions_of_positives_)
        bootstrap_ece.append(ece_)
        bootstrap_mce.append(mce_)

    frac_ci_low, frac_ci_high = np.nanquantile(bootstrap_fractions_of_positives, [alpha/2, 1 - alpha/2], axis=0)
    ece_ci_low, ece_ci_high = np.nanquantile(bootstrap_ece, [alpha/2, 1 - alpha/2])
    mce_ci_low, mce_ci_high = np.nanquantile(bootstrap_mce, [alpha/2, 1 - alpha/2])
    yerr = np.array([fractions_of_positives - frac_ci_low, frac_ci_high - fractions_of_positives])

    ax2 = ax.twinx() # create a second axis for the histogram

    # plot the histogram
    ax.hist(y_hats, bin_boundaries, facecolor='lightgrey', edgecolor='lightgrey', rwidth=0.9)

    # plot the diagonal (perfect calibarion) line
    ax2.plot([0, 1], [0, 1], linestyle='--', color="grey")

    # plot the mean calibrarion
    #ax2.plot(bin_centers, fractions_of_positives, "s-", label="calibration curve")
    ax2.errorbar(bin_centers, fractions_of_positives, yerr=yerr, fmt="s-", capsize=4,
            label=f'calibration curve (ECE {ece:.3f} ({ece_ci_low:.3f},{ece_ci_high:.3f}))')

    # manually set the plot range so that the zeros in y axes match up
    ax.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # swap the order of the tickmarks
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_label_position('right')
    ax2.yaxis.set_ticks_position('left')
    ax2.yaxis.set_label_position('left')

    # workaround for font size bug with twinx
    tick_label_fontsize = 8
    for label in ax2.get_yticklabels() + ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(tick_label_fontsize)

    # add grid
    ax.grid(True, axis='x', color='lightgrey', alpha=0.3)
    ax2.grid(True, axis='y', color='lightgrey', alpha=0.3)

    # add titles and legends
    if legend:
        ax2.legend(loc="lower right")
    else:
        ax2.text(0.11, 0.92, f'ECE {ece:.3f} ({ece_ci_low:.3f},{ece_ci_high:.3f})')
        ax2.text(0.11, 0.83, f'MCE {mce:.2f} ({mce_ci_low:.2f},{mce_ci_high:.2f})')

    if xlabel != '':
        ax.set(xlabel=xlabel)

    if len(ylabel)==2 and ylabel[0] != '':
        ax2.set(ylabel=ylabel[0])
    else:
        plt.setp(ax2.get_yticklabels(), visible=False)

    if len(ylabel)==2 and ylabel[1] != '':
        ax.set(ylabel=ylabel[1])
    else:
        plt.setp(ax.get_yticklabels(), visible=False)


def cv_calibration_plot(ax, ys, y_hats, n_bins=10, xlabel="Predicted Probability",
        ylabel="Observed Probability", legend=True):
    bin_boundaries = np.arange(n_bins + 1) / n_bins
    bin_boundaries[-1] = 1.0 + np.spacing(1.0) # make sure the final bin includes 1.0

    fractions_of_positives = []
    for y,y_hat in zip(ys,y_hats):
        bins = np.digitize(y_hat, bin_boundaries) - 1
        binpos = np.bincount(bins, y, minlength=n_bins)
        bincounts = np.bincount(bins, minlength=n_bins).astype(float)
        bincounts[bincounts == 0] = np.nan # empty bin means no valid data to average
        fractions_of_positives += [binpos / bincounts]

    bin_centers = (np.arange(n_bins) + 0.5) / n_bins

    # calculate some statistics, ignoring warnings for nans
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean_fraction_of_positives = np.nanmean(fractions_of_positives, axis=0)
        sd_fraction_of_positives = np.nanvar(fractions_of_positives, axis=0)**0.5
        mean_plus_sd = np.minimum(mean_fraction_of_positives + sd_fraction_of_positives, 1)
        mean_minus_sd = np.maximum(mean_fraction_of_positives - sd_fraction_of_positives, 0)

    # plot the grey +/- one SD range
    ax.fill_between(
        bin_centers, mean_plus_sd, mean_minus_sd,
        color="grey", alpha=0.2, label="mean calibration curve +/- 1 SD")

    # plot each fold
    first_fold = True
    for fraction_of_positives in fractions_of_positives:
        ax.plot(bin_centers, fraction_of_positives, "s-", color='blue', alpha=0.05,
            label="calibration curve for each fold" if first_fold else None)
        first_fold = False
    # plot the diagonal (perfect calibarion) line
    ax.plot([0, 1], [0, 1], linestyle='--', color="grey")

    # plot the mean calibrarion
    ax.plot(bin_centers, mean_fraction_of_positives, "s-", label="mean calibration curve")

    # add titles and legends
    if legend:
        ax.legend(loc="lower right")
    ax.set(xlabel=xlabel, ylabel=ylabel)


def parse_arguments():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('YS',
            type=argparse.FileType('rb'),
            help='bundle of actual y values for each fold')
    argument_parser.add_argument('YHATS',
            type=argparse.FileType('rb'),
            help='bundle of predicted y values for each fold')
    argument_parser.add_argument('PLOTFILE',
            help='filename to use to save the Calibration plot')
    argument_parser.add_argument('--title', type=str, default="Calibration")
    argument_parser.add_argument('--dpi', type=int, default=150)
    argument_parser.add_argument('--bootstraps', type=int, default=0,
            help='number of bootstraps to use for confidence intervals')

    return argument_parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    #print(args)

    print(f"{time.time() - startup_time}: loading ys")
    ys = pickle.load(args.YS)
    print(f"{time.time() - startup_time}: loading y_hats")
    y_hats = pickle.load(args.YHATS)

    print(f"{time.time() - startup_time}: generating plot")
    fig,ax = plt.subplots(figsize=(7,7))
    ax.set_title(args.title)
    if args.bootstraps > 0:
        bootstrapped_calibration_plot(ax, ys[0], y_hats[0], num_bootstraps=args.bootstraps)
    else:
        cv_calibration_plot(ax, ys, y_hats)
    plt.savefig(args.PLOTFILE, bbox_inches='tight', dpi=args.dpi)
