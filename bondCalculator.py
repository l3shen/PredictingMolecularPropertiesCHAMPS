import pandas as pd
import numpy as np
import itertools as it
import math
import time
import sys

def calcBondLengths(trainDataLocation, structureDataLocation, dipoleDataLocation, saveResults=True, verbose=True):

    # Load in data.
    trainDataRaw = pd.read_csv(trainDataLocation, header=0, index_col='id')
    structureDataRaw = pd.read_csv(structureDataLocation, header=0)
    dipoleData = pd.read_csv(dipoleDataLocation, header=0)

    print("Preparing data.")

    # Map atomic coordinates to dataset.
    trainData = map_atom_info(trainDataRaw, structureDataRaw, 0)
    trainData = map_atom_info(trainData, structureDataRaw, 1)

    print("Calculating bond lengths.")
    # Calculate bond length
    trainData['bond_dist'] = trainData.apply(lambda x: dist(x), axis=1)

    # Check to see if everything is a-OK.
    if verbose:
        print(trainData.head(20))

    # Save new structure data to csv file for ease of access later.
    # TODO: Save as HDF5? Pickle seems to run out of memory fast.

    if saveResults:
        print("Saving results to CSV file.")
        trainData.to_csv("trainDataPrepared.csv")

    return trainDataRaw


def map_atom_info(df, structureData, atom_idx):
    df = pd.merge(df, structureData, how='left',
                  left_on=['molecule_name', f'atom_index_{atom_idx}'],
                  right_on=['molecule_name', 'atom_index'])

    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

def dist(row):
    return ( (row['x_1'] - row['x_0'])**2 +
             (row['y_1'] - row['y_0'])**2 +
             (row['z_1'] - row['z_0'])**2 ) ** 0.5
