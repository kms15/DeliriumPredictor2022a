#!/usr/bin/python3

import numpy as np
import pandas as pd
import time
import json
import argparse

startup_time = time.time()

def parse_arguments():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('CSVFILE',
            type=argparse.FileType('rb'),
            help='files with the features at each snapshot')
    argument_parser.add_argument('NPYFILE',
            type=argparse.FileType('wb'),
            help='filename with which to save the npy version of the file')

    return argument_parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    #print(args)

    print(f"{time.time() - startup_time}: loading CSV")
    data = pd.read_csv(args.CSVFILE, parse_dates=["DTS"])

    print(f"{time.time() - startup_time}: saving NPY")
    np.save(args.NPYFILE, np.asarray(data.iloc[:,2:]))

    # generate the feature names
    #featurenames = [name for name in data.columns[2:]]
    #json.dump(featurenames, args.JSONFILE)
