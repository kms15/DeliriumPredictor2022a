#!/usr/bin/python3

import numpy as np
import scipy as sp
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
#from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import GroupKFold
#from sklearn.metrics import roc_curve, auc, roc_auc_score
#from sklearn.preprocessing import StandardScaler
#from sklearn.utils import resample
#from sklearn.calibration import calibration_curve, CalibratedClassifierCV
#import math
from sklearn.calibration import CalibratedClassifierCV
import time
#import joblib
import os.path
import warnings
#import feather
import pickle
import argparse
import json

startup_time = time.time()


def parse_arguments():
    argument_parser = argparse.ArgumentParser()
    subparsers = argument_parser.add_subparsers(
            dest='command', title='command')
    subparsers.required=True

    subparser_train = subparsers.add_parser('train',
            help='train the model on a given dataset')
    subparser_train.add_argument('DATAFILE', nargs='+',
            type=argparse.FileType('rb'),
            help='pairs of X and y datafiles used to train the model')
    subparser_train.add_argument('MODELFILE',
            help='filename to use to save the trained model\'s parameters')
    subparser_train.add_argument('--logl1penalty', default=-2.,
            help='log of L1 regularization penalty', type=float)

    subparser_apply = subparsers.add_parser('apply',
            help='apply the model to a given dataset')
    subparser_apply.add_argument('MODELFILE',
            type=argparse.FileType('rb'),
            help='file from which to load the model parameters')
    subparser_apply.add_argument('DATAFILE', nargs='+',
            type=argparse.FileType('rb'),
            help='file with inputs to use for the model')
    subparser_apply.add_argument('OUTFILE',
            help='file to save the predicted outputs')

    subparser_apply = subparsers.add_parser('applynames',
            help='apply the model to a given dataset')
    subparser_apply.add_argument('MODELFILE',
            type=argparse.FileType('rb'),
            help='file from which to load the model parameters')
    subparser_apply.add_argument('COLNAMES',
            type=argparse.FileType('r'),
            help='file with column names')
    subparser_apply.add_argument('OUTFILE',
            help='file to save the feature importances')

    return argument_parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    #print(args)

    if args.command == 'train':
        print(f"{time.time() - startup_time}: loading X")
        X_dataframes = [pd.read_csv(f, parse_dates=["DTS"])
                for f in args.DATAFILE[0::2]]
        X = np.vstack([np.asarray(df.iloc[:,2:]) for df in X_dataframes])
        del X_dataframes # these are large, so we try to free the memory
        print(f"{time.time() - startup_time}: loading y")
        y_dataframes = [pd.read_csv(f, parse_dates=["DTS"])
                for f in args.DATAFILE[1::2]]
        y = np.concatenate([np.asarray(df.iloc[:,2:])
            for df in y_dataframes]).ravel()

        print(f"{time.time() - startup_time}: creating model")
        l1penalty = 10**args.logl1penalty
        lr_model = (CalibratedClassifierCV(
                    LogisticRegression(
                        solver='saga', penalty='l1', C=l1penalty,
                        tol=1e-2, class_weight='balanced'),
                    method="sigmoid", cv=5)
                )

        print(f"{time.time() - startup_time}: fitting model")
        lr_model.fit(np.asarray(X), np.asarray(y))

        print(f"{time.time() - startup_time}: saving model")
        with open(args.MODELFILE, "wb") as f:
            pickle.dump(lr_model, f)

    elif args.command == 'apply':
        print(f"{time.time() - startup_time}: loading X")
        X_dataframes = [pd.read_csv(f, parse_dates=["DTS"])
                for f in args.DATAFILE]
        X = np.vstack([np.asarray(df.iloc[:,2:]) for df in X_dataframes])
        mrndts = pd.concat([df[["MRN","DTS"]] for df in X_dataframes],
                axis=0).reset_index(drop=True)
        del X_dataframes # these are large, so we try to free the memory
        print(f"{time.time() - startup_time}: loading model")
        lr_model = pickle.load(args.MODELFILE)
        print(f"{time.time() - startup_time}: applying model")
        y_hat = lr_model.predict_proba(np.asarray(X))[:, 1]
        print(f"{time.time() - startup_time}: saving result")
        y_hat_dataframe = pd.concat([mrndts, pd.DataFrame(dict(y_hat=y_hat))],
                axis=1)
        y_hat_dataframe.to_csv(args.OUTFILE, index=False)

    elif args.command == 'applynames':
        print(f"{time.time() - startup_time}: loading model")
        lr_model = pickle.load(args.MODELFILE)
        print(f"{time.time() - startup_time}: loading columns")
        colnames = json.load(args.COLNAMES)
        print(f"{time.time() - startup_time}: saving result")
        #importances = [c.base_estimator.feature_importances_
        #        for c in rf_model.calibrated_classifiers_]
        #mean_importances = np.mean(np.asarray(importances), axis=0)
        #importances = np.abs(lr_model.coef_[0,:])
        coefs = np.mean([m.base_estimator.coef_[0,:]
            for m in lr_model.calibrated_classifiers_], axis=0)
        importances = np.abs(coefs)
        signs = np.sign(coefs)
        with open(args.OUTFILE, "w") as f:
            f.write("feature,importance,sign\n")
            for feature,importance,sign in zip(colnames, importances, signs):
                f.write(f"{feature},{importance:.9f},{sign}\n")
