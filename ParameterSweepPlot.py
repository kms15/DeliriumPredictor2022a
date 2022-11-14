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

startup_time = time.time()

def parameter_sweep_plot(paramvals, aucs):
    mean_auc = np.mean(aucs, axis=1)
    sd_auc = (np.var(aucs, axis=1))**0.5

    # plot the grey +/- one SD range
    plt.fill_between(paramvals, np.minimum(mean_auc + sd_auc, 1), np.maximum(mean_auc - sd_auc, 0),
        color="grey", alpha=0.2, label="mean AUC +/- 1 SD")
    # plot the AUC for each fold
    first_fold = True
    for auc in aucs.transpose():
        plt.plot(paramvals, auc, color='blue', alpha=0.05,
            label="AUC for each fold in cross validation" if first_fold else None)
        first_fold = False
    # plot the mean AUC
    plt.plot(paramvals, mean_auc, label="mean AUC")

    # add titles and legends
    plt.legend(loc="lower right")
    plt.xlabel("Log Lambda")
    plt.ylabel("AUC")


def parse_arguments():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('INPUTS', nargs='+',
            help='pairs of x-values and auc datafiles')
    argument_parser.add_argument('PLOTFILE',
            help='filename to use to save the ROC plot')
    argument_parser.add_argument('--title', type=str, default="ROC")
    argument_parser.add_argument('--dpi', type=int, default=150)

    return argument_parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    #print(args)

    paramvals = np.asarray([float(val) for val in args.INPUTS[0::2]])

    print(f"{time.time() - startup_time}: loading aucs")
    aucs = np.asarray([np.loadtxt(filename) for filename in args.INPUTS[1::2]])

    print(f"{time.time() - startup_time}: generating plot")
    plt.figure(figsize=(7,7))
    plt.title(args.title)
    parameter_sweep_plot(paramvals, aucs)
    plt.savefig(args.PLOTFILE, bbox_inches='tight', dpi=args.dpi)
