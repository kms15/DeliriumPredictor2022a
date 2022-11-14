#!/usr/bin/python3

import numpy as np
import pandas as pd
import time
import argparse

startup_time = time.time()

def parse_arguments():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('CAMFILE',
            type=argparse.FileType('rb'),
            help='file with the CAM results to predict')
    argument_parser.add_argument('GROUPS',
            type=argparse.FileType('rb'),
            help='file with the holdout and CV group for each MRN')
    argument_parser.add_argument('TARGETPREFIX',
            help='prefix to add to output files')
    argument_parser.add_argument('--useholdout',
            help='use only the holdout set (otherwise exclude the holdout set)',
            action='store_true')

    return argument_parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    #print(args)

    print(f"{time.time() - startup_time}: loading y")
    ytable = pd.read_csv(args.CAMFILE, parse_dates=["DTS"],
            dtype={"MRN":np.int64})
    print(f"{time.time() - startup_time}: loading groups")
    grouptable = pd.read_parquet(args.GROUPS)


    print(f"{time.time() - startup_time}: calculating groups")
    mrn_to_holdout = dict(zip(grouptable['MRN'],grouptable['Holdout']))
    mrn_to_cvgroup = dict(zip(grouptable['MRN'],grouptable['CVGroup']))
    num_cv_groups = max(grouptable['CVGroup']) + 1

    # older data has MRNs with CAM screens but no demographics
    # once this is fixed we should remove the missing demographics
    holdouts_alldata = ytable['MRN'].map(lambda mrn: mrn_to_holdout[mrn]
        if mrn in mrn_to_holdout else True)
    cv_groups_alldata = ytable['MRN'].map(lambda mrn: mrn_to_cvgroup[mrn]
        if mrn in mrn_to_cvgroup else 0)
    missing_demographics = ytable['MRN'].map(lambda mrn: not mrn in mrn_to_holdout)


    print(f"{time.time() - startup_time}: generating X and y")

    # delirium is true if either of the CAM screens were positive
    y_raw = pd.DataFrame({'MRN':ytable['MRN'], 'DTS':ytable['DTS'],
            'y':ytable['CAM_max'] > 0})

    # remove the holdout groups
    if args.useholdout:
        y = y_raw.loc[holdouts_alldata & ~missing_demographics]
        cvgroups = cv_groups_alldata[holdouts_alldata & ~missing_demographics]
    else:
        y = y_raw.loc[~holdouts_alldata & ~missing_demographics]
        cvgroups = cv_groups_alldata[~holdouts_alldata & ~missing_demographics]

    print(f"{time.time() - startup_time}: saving folds")
    for i in range(num_cv_groups):
        #np.save(f'{args.TARGETPREFIX}y_fold{i}.npy', y[cvgroups == i])
        pd.DataFrame(y.loc[cvgroups == i]).to_csv(
                f'{args.TARGETPREFIX}y_fold{i}.csv',
                index=False, date_format="%Y-%m-%dT%H:%M:%S")
