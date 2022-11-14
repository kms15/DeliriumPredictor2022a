#!/usr/bin/python3

import numpy as np
import sklearn
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.constraints import maxnorm
import time
import os.path
import warnings
import pickle
import argparse
import json
import pandas as pd

startup_time = time.time()

# limit GPU memory used
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

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
            help='file from which to load the model parameters')
    subparser_apply.add_argument('DATAFILE', nargs='+',
            type=argparse.FileType('rb'),
            help='file with inputs to use for the model')
    subparser_apply.add_argument('OUTFILE',
            help='file to save the predicted outputs')

    return argument_parser.parse_args()

def build_fn(input_dim, hidden_dim=8):
    model = Sequential()
    model.add(Dense(units=4*hidden_dim, activation='relu', kernel_constraint=maxnorm(3), input_dim=input_dim))
    model.add(Dropout(0.5))
    model.add(Dense(units=2*hidden_dim, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(units=hidden_dim, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid', kernel_constraint=maxnorm(3)))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

if __name__ == "__main__":

    args = parse_arguments()

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

        # weight the classes by scarcity
        fraction_true = np.sum(y)/y.shape[0]
        print(f"{fraction_true * 100}% of training samples are positive...\n")
        class_weight = { 0:fraction_true, 1:(1-fraction_true) }

        print(f"{time.time() - startup_time}: creating model")
        ann_model = build_fn(X.shape[1])

        print(f"{time.time() - startup_time}: fitting model")
        ann_model.fit(np.asarray(X), np.asarray(y), batch_size=128,
                class_weight=class_weight, epochs=40, verbose=2)
        ann_model.fit(np.asarray(X), np.asarray(y), batch_size=128,
                #class_weight=class_weight,
                epochs=10, verbose=2)

        print(f"{time.time() - startup_time}: saving model")
        ann_model.save(args.MODELFILE)

    elif args.command == 'apply':
        print(f"{time.time() - startup_time}: loading X")
        X_dataframes = [pd.read_csv(f, parse_dates=["DTS"])
                for f in args.DATAFILE]
        X = np.vstack([np.asarray(df.iloc[:,2:]) for df in X_dataframes])
        mrndts = pd.concat([df[["MRN","DTS"]] for df in X_dataframes],
                axis=0).reset_index(drop=True)
        del X_dataframes # these are large, so we try to free the memory
        print(f"{time.time() - startup_time}: loading model")
        ann_model = load_model(args.MODELFILE)
        print(f"{time.time() - startup_time}: applying model")
        y_hat = ann_model.predict(np.asarray(X))[:, 0]
        print(f"{time.time() - startup_time}: saving result")
        y_hat_dataframe = pd.concat([mrndts, pd.DataFrame(dict(y_hat=y_hat))],
                axis=1)
        y_hat_dataframe.to_csv(args.OUTFILE, index=False)
