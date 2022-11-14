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

startup_time = time.time()

def bootstrapped_roc_plot(ax, ys, y_hats, xlabel="False Positive Rate",
        ylabel="True Positive Rate", legend=True,
        num_bootstraps=1000):

    alpha = 0.05 # 1 - (confidence interval = 95%)

    # generate the data for the main ROC curve
    fpr, tpr, thresh = roc_curve(ys, y_hats)
    main_auc = auc(fpr, tpr)

    # generate the bootstrap samples
    bootstrap_samples = [resample(ys, y_hats, random_state=i) for i in range(num_bootstraps)]

    # generate the bootstrap curves
    bootstrapped_roc_curves = [roc_curve(y, y_hat) for y, y_hat
        in bootstrap_samples]

    # interpolate the bootstrap curves onto a single x-axis so that we can
    # then calculate some statistics on the y-axis
    interp_fpr = np.linspace(0., 1., 1000)
    interp_tpr_bootstraps = np.array([np.interp(interp_fpr, fpr, tpr)
        for fpr, tpr, thresh in bootstrapped_roc_curves])
    roc_ci_low, roc_ci_high = np.quantile(
        interp_tpr_bootstraps, [alpha/2, 1 - alpha/2], axis=0)
    aucs = [auc(fpr, tpr) for fpr, tpr, thresh in bootstrapped_roc_curves]
    auc_ci_low, auc_ci_high = np.quantile(
        aucs, [alpha/2, 1 - alpha/2], axis=0)

    # plot the grey +/- one SD range
    ax.fill_between(interp_fpr, roc_ci_low, roc_ci_high,
            color="grey", alpha=0.2, label=f'{100*(1 - alpha):.0f}% confidence interval')
    # plot the diagonal (zero information) line
    ax.plot([0, 1], [0, 1], linestyle='--', color="grey")
    # plot the mean ROC
    ci_precision = 2 if auc_ci_high - auc_ci_low > 0.02 else 3
    ax.plot(fpr, tpr,
            label=f'ROC (AUROC {main_auc:.2f} ({auc_ci_low:.{ci_precision}f},{auc_ci_high:.{ci_precision}f}))')
    #ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)

    # add grid
    ax.grid(True, color = (0.95, 0.95, 0.95))

    # add titles and legends
    if legend:
        ax.legend(loc="lower right")
    else:
        ax.text(0.1, 0.0, f'AUROC {main_auc:.2f} ({auc_ci_low:.{ci_precision}f},{auc_ci_high:.{ci_precision}f})')
    #ax.set_xlim(0, 1)
    ax.set(xlabel=xlabel, ylabel=ylabel)


def cv_roc_plot(ax, ys, y_hats, xlabel="False Positive Rate",
        ylabel="True Positive Rate", legend=True,
        num_bootstraps=0):
    # generate the roc curves for each fold
    roc_curves = [roc_curve(y, y_hat) for y, y_hat in zip(ys, y_hats)]

    # interpolate the curves onto a single x-axis so that we can then calculate some statistics on
    # the y-axis
    interp_fpr = np.linspace(0., 1., 1000)
    interp_tpr_cvs = np.array([np.interp(interp_fpr, fpr, tpr) for fpr, tpr, thresh in roc_curves])
    mean_tpr = np.mean(interp_tpr_cvs, axis=0)
    sd_tpr = (np.var(interp_tpr_cvs, axis=0))**0.5
    aucs = [auc(fpr, tpr) for fpr, tpr, thresh in roc_curves]

    # plot the grey +/- one SD range
    ax.fill_between(interp_fpr, np.minimum(mean_tpr + sd_tpr, 1), np.maximum(mean_tpr - sd_tpr, 0),
        color="grey", alpha=0.2, label="mean ROC +/- 1 SD")
    # plot the ROC for each fold
    first_fold = True
    for fpr, tpr, thresh in roc_curves:
        ax.plot(fpr, tpr, color='blue', alpha=0.05,
            label="ROC for each fold in cross validation" if first_fold else None)
        first_fold = False
    # plot the diagonal (zero information) line
    ax.plot([0, 1], [0, 1], linestyle='--', color="grey")
    # plot the mean ROC
    ax.plot(interp_fpr, mean_tpr,
        label="mean ROC (AUC {0:.2f}, SD {1:.3f})".format(np.mean(aucs), np.var(aucs)**0.5))
    #ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)

    # add titles and legends
    if legend:
        ax.legend(loc="lower right")
    else:
        ax.text(0.6, 0.0, f'AUC {np.mean(aucs):0.2}')
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
            help='filename to use to save the ROC plot')
    argument_parser.add_argument('--title', type=str, default="ROC")
    argument_parser.add_argument('--dpi', type=int, default=150)
    argument_parser.add_argument('--bootstraps', type=int, default=0,
            help='number of bootstraps to use for confidence intervals')

    return argument_parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    print(f"{time.time() - startup_time}: loading ys")
    ys = pickle.load(args.YS)
    print(f"{time.time() - startup_time}: loading y_hats")
    y_hats = pickle.load(args.YHATS)

    print(f"{time.time() - startup_time}: generating plot")
    fig,ax = plt.subplots(figsize=(7,7))
    ax.set_title(args.title)
    if args.bootstraps > 0:
        bootstrapped_roc_plot(ax, ys[0], y_hats[0], num_bootstraps=args.bootstraps)
    else:
        cv_roc_plot(ax, ys, y_hats)
    plt.savefig(args.PLOTFILE, bbox_inches='tight', dpi=args.dpi)
