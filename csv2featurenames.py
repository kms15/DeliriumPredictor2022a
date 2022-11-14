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
    argument_parser.add_argument('FEATURENAMES',
            type=argparse.FileType('w'),
            help='JSON file to create with the featurenames')

    return argument_parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    #print(args)

    print(f"{time.time() - startup_time}: loading CSV")
    data = pd.read_csv(args.CSVFILE, parse_dates=["DTS"], nrows=1)

    # generate the feature names
    print(f"{time.time() - startup_time}: saving names")
    featurenames = [name for name in data.columns[2:]]
    json.dump(featurenames, args.FEATURENAMES)
