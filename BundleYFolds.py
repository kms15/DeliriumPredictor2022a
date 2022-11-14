#!/usr/bin/python3

import pandas as pd
import numpy as np
import time
import argparse
import pickle

startup_time = time.time()

def parse_arguments():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('YFILE', nargs='+',
            type=argparse.FileType('rb'),
            help='individual y files to bundle together')
    argument_parser.add_argument('BUNDLEFILE',
            help='filename of the generated y datafile bundle')

    return argument_parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    print(f"{time.time() - startup_time}: loading ys")
    y_dataframes = [pd.read_csv(f, parse_dates=["DTS"]) for f in args.YFILE]
    ys = [np.asarray(df.iloc[:,2:]).ravel() for df in y_dataframes]

    print(f"{time.time() - startup_time}: generating bundle")
    with open(args.BUNDLEFILE, 'wb') as f:
        pickle.dump(ys, f)
