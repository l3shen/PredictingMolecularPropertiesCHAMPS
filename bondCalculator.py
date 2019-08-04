import pandas as pd
import numpy as np
import itertools as it
import math
import time
import sys

def calcBondLengths(dataLocation, structureDataLocation, dataSetType, saveResults=True, verbose=True):

    # Load in data.
    dataRaw = pd.read_csv(dataLocation, header=0)
    structureDataRaw = pd.read_csv(structureDataLocation, header=0)
    # dipoleData = pd.read_csv(dipoleDataLocation, header=0)

    print("Preparing data.")

    # Map atomic coordinates to dataset.
    procData = map_atom_info(dataRaw, structureDataRaw, 0)
    procData = map_atom_info(procData, structureDataRaw, 1)

    print("Calculating bond lengths.")
    # Calculate bond length
    procData['bond_dist'] = procData.apply(lambda x: dist(x), axis=1)

    # Convert type data to generic integer format.
    # trainData['type'].replace(TYPE_DICT, inplace=True)

    # Check to see if everything is a-OK.
    if verbose:
        print(procData.head(20))

    # Save new structure data to csv file for ease of access later.
    # TODO: Save as HDF5? Pickle seems to run out of memory fast.

    if saveResults and dataSetType == 'train':
        print("Saving train results to CSV file.")
        procData.to_csv("trainDataPrepared.csv")
        print("Saved.")
    elif saveResults and dataSetType == 'test':
        print("Saving test results to CSV file.")
        procData.to_csv("testDataPrepared.csv")
        print("Saved.")

    return dataRaw


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




