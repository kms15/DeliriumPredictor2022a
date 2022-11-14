#!/usr/bin/python3

import numpy as np
import scipy as sp
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.linear_model import LogisticRegression
from sklearn.svm import NuSVC
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
        svc_model = (#CalibratedClassifierCV(
                    NuSVC(nu=0.1, class_weight='balanced', random_state=1811,
                        probability=True, gamma='auto', max_iter=100)#,
                #    method="sigmoid", cv=5
                )

        print(f"{time.time() - startup_time}: fitting model")
        svc_model.fit(np.asarray(X), np.asarray(y))

        print(f"{time.time() - startup_time}: saving model")
        with open(args.MODELFILE, "wb") as f:
            pickle.dump(svc_model, f)

    elif args.command == 'apply':
        print(f"{time.time() - startup_time}: loading X")
        X_dataframes = [pd.read_csv(f, parse_dates=["DTS"])
                for f in args.DATAFILE]
        X = np.vstack([np.asarray(df.iloc[:,2:]) for df in X_dataframes])
        mrndts = pd.concat([df[["MRN","DTS"]] for df in X_dataframes],
                axis=0).reset_index(drop=True)
        del X_dataframes # these are large, so we try to free the memory
        print(f"{time.time() - startup_time}: loading model")
        svc_model = pickle.load(args.MODELFILE)
        print(f"{time.time() - startup_time}: applying model")
        y_hat = svc_model.predict_proba(np.asarray(X))[:, 1]
        print(f"{time.time() - startup_time}: saving result")
        y_hat_dataframe = pd.concat([mrndts, pd.DataFrame(dict(y_hat=y_hat))],
                axis=1)
        y_hat_dataframe.to_csv(args.OUTFILE, index=False)

