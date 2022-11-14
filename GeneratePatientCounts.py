#!/usr/bin/python3

import numpy as np
import pandas as pd
import time
import argparse

startup_time = time.time()

def parse_arguments():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('Y_TRAIN',
            type=argparse.FileType('rb'),
            help='csv file with training labels')
    argument_parser.add_argument('Y_HOLDOUT',
            type=argparse.FileType('rb'),
            help='csv file with the holdout labels')
    argument_parser.add_argument('OUTPUT',
            type=argparse.FileType('w'),
            help='csv file to generate with the output statistics')

    return argument_parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    y_train = pd.read_csv(args.Y_TRAIN, parse_dates=["DTS"],
            dtype={"MRN":np.int64})
    y_holdout = pd.read_csv(args.Y_HOLDOUT, parse_dates=["DTS"],
            dtype={"MRN":np.int64})

    args.OUTPUT.write("statistic,value\n")
    args.OUTPUT.write(f'num_patients_train,{y_train.MRN.nunique()}\n')
    args.OUTPUT.write(f'num_patients_holdout,{y_holdout.MRN.nunique()}\n')
    args.OUTPUT.write(f'num_patients_total,{y_train.MRN.nunique() + y_holdout.MRN.nunique()}\n')
    args.OUTPUT.write(f'num_days_train,{y_train.shape[0]}\n')
    args.OUTPUT.write(f'num_days_holdout,{y_holdout.shape[0]}\n')
    args.OUTPUT.write(f'num_days_total,{y_train.shape[0] + y_holdout.shape[0]}\n')
