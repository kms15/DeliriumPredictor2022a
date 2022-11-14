#!/usr/bin/python3
import numpy as np
import scipy as sp
import pandas as pd
import argparse

import math
import hashlib
import time
import os.path

def randomizationTextForPatient(patient):
    firstName = patient.FullName.split()[0].upper();
    lastName = patient.FullName.split()[-1].upper();
    return '|'.join([firstName, lastName, patient.Gender, str(patient.BirthDTS)])

def uniform_deviate_from_text(text):
    m = hashlib.sha256()
    m.update(bytes(text.encode()))
    s = m.digest()
    bits_per_byte = 8
    bytes_used = 8
    return (sum([s[i] << (bits_per_byte*i) for i in range(bytes_used)]) /
            (math.pow(2, bytes_used*bits_per_byte) - 1))

def RandomizationValueForPatient(patient):
    return uniform_deviate_from_text(randomizationTextForPatient(patient))

startup_time = time.time()

def parse_arguments():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('Patients',
            type=argparse.FileType('r'),
            help='csv file with patient data')
    argument_parser.add_argument('csvoutput',
            help='target csv file')
    argument_parser.add_argument('snappyoutput',
            help='target snappy file')

    return argument_parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    #print(args)

    print(f"{time.time() - startup_time}: loading patients")

    holdoutFraction = 0.2
    numCVGroups = 10;

    patients = pd.read_csv(args.Patients, parse_dates=['BirthDTS'], dtype={'MRN' : int})
    patients['RandomizationValue'] = patients.apply(RandomizationValueForPatient, axis=1)
    patients['Holdout'] = (patients.RandomizationValue < holdoutFraction)

    randomizationValueAfterHoldout = (patients.RandomizationValue - holdoutFraction) / (1 - holdoutFraction)
    randomizationValueAfterHoldout[patients.Holdout] = patients.RandomizationValue / holdoutFraction

    patients['CVGroup'] = (numCVGroups * randomizationValueAfterHoldout).astype(np.int)

    patients[['MRN','Holdout','CVGroup']].to_parquet(args.snappyoutput)
    patients[['MRN','Holdout','CVGroup']].to_csv(args.csvoutput, index=False)
