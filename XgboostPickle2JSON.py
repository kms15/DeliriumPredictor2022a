#!/usr/bin/env python3

import numpy as np
import xgboost as xgb
import pickle
import argparse

def parse_arguments():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('PICKLEFILE',
            type=argparse.FileType('rb'),
            help='pickle file from which to load the model parameters')
    argument_parser.add_argument('OUTFILE',
            help='JSON file to save the model parameters')

    return argument_parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    bt_model = pickle.load(args.PICKLEFILE)
    bt_model.save_model(args.OUTFILE)
