#!/usr/bin/python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import precision_recall_curve, auc
from sklearn.utils import resample
import time
import os.path
import argparse
import pickle

startup_time = time.time()

def bootstrapped_pr_plot(ax, ys, y_hats, xlabel='Recall (true positive rate)', 
        ylabel='Precision (positive predictive value)', legend=True,
        num_bootstraps=1000):

    alpha = 0.05 # 1 - (confidence interval = 95%)

    # generate the data for the main PR curve
    precision, recall, thresh = precision_recall_curve(ys, y_hats)
    main_auc = auc(recall, precision)

    # generate the bootstrap samples
    bootstrap_samples = [resample(ys, y_hats) for i in range(num_bootstraps)]

    # generate the bootstrap curves
    bootstrapped_pr_curves = [precision_recall_curve(y, y_hat) for y, y_hat
        in bootstrap_samples]

    # interpolate the bootstrap curves onto a single x-axis so that we can
    # then calculate some statistics on the y-axis
    interp_recall = np.linspace(0., 1., 1000)
    interp_precision_bootstraps = np.array([np.interp(interp_recall, recall[::-1], precision[::-1])
        for precision, recall, thresh in bootstrapped_pr_curves])
    pr_ci_low, pr_ci_high = np.quantile(
        interp_precision_bootstraps, [alpha/2, 1 - alpha/2], axis=0)
    aucs = [auc(recall, precision) for precision, recall, thresh in bootstrapped_pr_curves]
    auc_ci_low, auc_ci_high = np.quantile(
        aucs, [alpha/2, 1 - alpha/2], axis=0)
    ci_precision = 2 if auc_ci_high - auc_ci_low > 0.02 else 3

    # plot the grey +/- one SD range
    ax.fill_between(interp_recall, pr_ci_low, pr_ci_high,
            color="grey", alpha=0.2, label=f'{100*(1 - alpha):.0f}% confidence interval')
    # plot the mean PR
    ax.plot(recall, precision,
            label=f'PR curve (AUPRC {main_auc:.2f} ({auc_ci_low:.{ci_precision}f},{auc_ci_high:.{ci_precision}f})) (Incidence: {np.mean(ys):.2f})')
    #ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)

    # add grid
    ax.grid(True, color = (0.95, 0.95, 0.95))

    # add titles and legends
    if legend:
        ax.legend(loc="lower right")
    else:
        ax.text(0.0, 0.0, f'AUPRC {main_auc:.2f} ({auc_ci_low:.{ci_precision}f},{auc_ci_high:.{ci_precision}f})')
    ax.set(xlabel=xlabel, ylabel=ylabel)


def cv_pr_plot(ax, ys, y_hats, xlabel='Recall (true positive rate)', 
        ylabel='Precision (positive predictive value)', legend=True):
    # generate the PR curves for each fold
    pr_curves = [precision_recall_curve(y, y_hat) for y, y_hat in zip(ys, y_hats)]

    # interpolate the curves onto a single x-axis so that we can then calculate some statistics on
    # the y-axis
    interp_recall = np.linspace(0., 1., 1000)
    interp_precision_cvs = np.array([np.interp(interp_recall, recall[::-1], precision[::-1])
        for precision, recall, thresh in pr_curves])
    mean_precision = np.mean(interp_precision_cvs, axis=0)
    sd_precision = (np.var(interp_precision_cvs, axis=0))**0.5
    aucs = [auc(recall, precision) for precision, recall, thresh in pr_curves]

    # plot the grey +/- one SD range
    ax.fill_between(interp_recall, np.minimum(mean_precision + sd_precision, 1), np.maximum(mean_precision - sd_precision, 0),
        color="grey", alpha=0.2, label="mean PR +/- 1 SD")
    # plot the PR curve for each fold
    first_fold = True
    for precision, recall, thresh in pr_curves:
        ax.plot(recall, precision, color='blue', alpha=0.05,
            label="PR for each fold in cross validation" if first_fold else None)
        first_fold = False
    # plot the mean PR
    ax.plot(interp_recall, mean_precision,
        label="mean PR (AUC {0:.3f}, SD {1:.3f})".format(np.mean(aucs), np.var(aucs)**0.5))

    # add titles and legends
    if legend:
        ax.legend(loc="lower right")
    else:
        ax.text(0.1, 0., f'AUC {np.mean(aucs):0.3}')
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
            help='filename to use to save the PR plot')
    argument_parser.add_argument('--title', type=str, default="Precision-Recall")
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
        bootstrapped_pr_plot(ax, ys[0], y_hats[0], num_bootstraps=args.bootstraps)
    else:
        cv_pr_plot(ax, ys, y_hats)
    plt.savefig(args.PLOTFILE, bbox_inches='tight', dpi=args.dpi)
