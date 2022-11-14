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

def calculate_AUCs(ys, y_hats):
    # generate the roc curves for each fold
    roc_curves = [roc_curve(y, y_hat) for y, y_hat in zip(ys, y_hats)]

    # interpolate the curves onto a single x-axis so that we can then calculate some statistics on
    # the y-axis
    interp_fpr = np.linspace(0., 1., 1000)
    interp_tpr_cvs = np.array([sp.interp(interp_fpr, fpr, tpr) for fpr, tpr, thresh in roc_curves])
    mean_tpr = np.mean(interp_tpr_cvs, axis=0)
    sd_tpr = (np.var(interp_tpr_cvs, axis=0))**0.5
    aucs = [auc(fpr, tpr) for fpr, tpr, thresh in roc_curves]

    return aucs

def parse_arguments():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('DATAFILE', nargs='+',
            type=argparse.FileType('rb'),
            help='pairs of y and y_hat datafiles')
    argument_parser.add_argument('AUCFILE',
            help='filename to use to save the aucs')

    return argument_parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    print(f"{time.time() - startup_time}: loading ys")
    y_dataframes = [pd.read_csv(f, parse_dates=["DTS"])
            for f in args.DATAFILE[0::2]]
    ys = [np.asarray(df.iloc[:,2:]).ravel() for df in y_dataframes]
    print(f"{time.time() - startup_time}: loading y_hats")
    y_hat_dataframes = [pd.read_csv(f, parse_dates=["DTS"])
            for f in args.DATAFILE[1::2]]
    y_hats = [np.asarray(df.iloc[:,2:]).ravel() for df in y_hat_dataframes]

    print(f"{time.time() - startup_time}: generating summary")
    aucs = calculate_AUCs(ys, y_hats)
    np.savetxt(args.AUCFILE, aucs)
