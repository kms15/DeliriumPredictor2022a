#!/usr/bin/env python3

import xgboost as xgb
import shap
import argparse
import pandas as pd
from matplotlib import pyplot as plt

def parse_arguments():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('MODELJSON',
            help='JSON file from which to load the model parameters')
    argument_parser.add_argument('XVALS',
            help='CSV file of inputs examples to use for the SHAP plot')
    argument_parser.add_argument('PLOTFILE',
            help='PNG file to which to save the output')
    argument_parser.add_argument('--dpi', type=int, default=150)

    return argument_parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    bt_model = xgb.XGBClassifier()
    bt_model.load_model(args.MODELJSON)

    X = pd.read_csv(args.XVALS, parse_dates=["DTS"]).iloc[:,2:]

    explainer = shap.explainers.Tree(bt_model, feature_names=X.columns)
    shap_values = explainer(X.values)

    plt.figure(figsize=(15,20))
    shap.plots.beeswarm(shap_values, max_display=41, plot_size=None)
    plt.savefig(args.PLOTFILE, bbox_inches='tight', dpi=args.dpi)
