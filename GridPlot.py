#!/usr/bin/python3

import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import roc_curve, auc, roc_auc_score
import time
import os.path
import argparse
import pickle
import sys
from ROCPlot import cv_roc_plot, bootstrapped_roc_plot
from PRPlot import cv_pr_plot, bootstrapped_pr_plot
from CalibrationPlot import cv_calibration_plot, bootstrapped_calibration_plot

startup_time = time.time()


def parse_arguments():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--rownames', nargs='+', required=True,
            help='names for each of the rows')
    argument_parser.add_argument('--colnames', nargs='+', required=True,
            help='names for each of the columns')
    argument_parser.add_argument('--ys', nargs='+', required=True,
            type=argparse.FileType('rb'),
            help='bundles of actual y values for each grid square')
    argument_parser.add_argument('--yhats', nargs='+', required=True,
            type=argparse.FileType('rb'),
            help='bundles of predicted y values for each grid square')
    argument_parser.add_argument('--out',
            help='filename to use to save the grid plot')
    argument_parser.add_argument('--dpi', type=int, default=150,
            help='resolution of the final plot (dots per inch)')
    argument_parser.add_argument('--bootstraps', type=int, default=0,
            help='number of bootstraps to use for the confidence interval')
    argument_parser.add_argument('--precisionrecall', action='store_true',
            help='plot precision-recall curves instead of ROC curves')
    argument_parser.add_argument('--calibration', action='store_true',
            help='plot calibration curves instead of ROC curves')

    return argument_parser.parse_args()


labelfont = {
    'family': 'sans',
    'weight': 'normal',
    'size': 14,
}


if __name__ == "__main__":

    args = parse_arguments()
    #print(args)

    nrows = len(args.rownames)
    ncols = len(args.colnames)
    if len(args.ys) != nrows * ncols:
        print(f"Error: expected {nrows * ncols} ys, but received {len(args.ys)}")
        sys.exit(1)
    if len(args.yhats) != nrows * ncols:
        print(f"Error: expected {nrows * ncols} yhats, but received {len(args.yhats)}")
        sys.exit(1)

    print(f"{time.time() - startup_time}: loading ys")
    ys = np.array([pickle.load(y) for y in args.ys], dtype=object).reshape(nrows, ncols, -1)
    print(f"{time.time() - startup_time}: loading y_hats")
    y_hats = np.array([pickle.load(yhat) for yhat in args.yhats], dtype=object).reshape(nrows, ncols, -1)

    print(f"{time.time() - startup_time}: generating plot")


    if args.precisionrecall:
        xlabel='Recall'
        ylabel='Precision'
        ylabel2=''
    elif args.calibration:
        xlabel='Predicted Probability'
        ylabel=('Observed Probability', '')
        ylabel2=('', 'Predictions in bin')
    else:
        xlabel='False positive rate'
        ylabel='True positive rate'
        ylabel2=''

    if args.bootstraps == 0:
        if args.precisionrecall:
            plotfunc = cv_pr_plot
        elif args.calibration:
            plotfunc = cv_calibration_plot
        else:
            plotfunc = cv_roc_plot
    else:
        if args.precisionrecall:
            plotfunc = lambda ax, ys, y_hats, xlabel, ylabel,legend : (
                bootstrapped_pr_plot(ax, ys[0], y_hats[0], xlabel, ylabel,
                legend=legend, num_bootstraps=args.bootstraps))
        elif args.calibration:
            plotfunc = lambda ax, ys, y_hats, xlabel, ylabel,legend : (
                bootstrapped_calibration_plot(ax, ys[0], y_hats[0], xlabel, ylabel,
                legend=legend, num_bootstraps=args.bootstraps))
        else:
            plotfunc = lambda ax, ys, y_hats, xlabel, ylabel,legend : (
                bootstrapped_roc_plot(ax, ys[0], y_hats[0], xlabel, ylabel,
                legend=legend, num_bootstraps=args.bootstraps))

    fig,axs = plt.subplots(nrows, ncols, figsize=(3*ncols,3*nrows), sharex=True, sharey=True,
            gridspec_kw=dict(wspace=0.1, hspace=0.1))
    for i in range(nrows):
        for j in range(ncols):
            plotfunc(axs[i,j],ys[i,j], y_hats[i,j],
                    xlabel=xlabel if i == nrows-1 else '',
                    ylabel=ylabel if j == 0 else ylabel2 if j == ncols-1 else '',
                    legend=False)
            axs[i,j].tick_params(labelsize="small")
            axs[i,j].spines['top'].set_visible(False)
            axs[i,j].spines['right'].set_visible(False)
    for j in range(ncols):
        #plt.figtext(0.255 + 0.265*j, 0.90, args.colnames[j], ha="center", rotation=0, fontdict=labelfont)
        #axs[-1,j].set_xlabel(xlabel)
        axs[0,j].set_title(args.colnames[j], fontdict=labelfont)
    for i in range(nrows):
        # HACK: hand-coding the positions of the labels here is fragile, but
        # probably as maintainable as the many lines of code required to do an
        # axes to figure coordinate transformation in matplotlib.
        plt.figtext(0.045, 0.198 + 0.197*(nrows - i - 1), args.rownames[i],
                ha="center", va="center", rotation=90, fontdict=labelfont)
        #axs[i,0].set_ylabel(ylabel)
    plt.savefig(args.out, bbox_inches='tight', dpi=args.dpi)
