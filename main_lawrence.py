import pandas as pd
import numpy as np
import itertools as it
import math
import time


def map_atom_info(df, atom_idx):
    df = pd.merge(df, structureDataRaw, how='left',
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

def mst_matrix(index):
    matrix = np.array([[mag_shield_tensorData.iloc[index, 2], mag_shield_tensorData.iloc[index, 3], mag_shield_tensorData.iloc[index, 4]],
                    [mag_shield_tensorData.iloc[index, 5], mag_shield_tensorData.iloc[index, 6], mag_shield_tensorData.iloc[index, 7]],
                    [mag_shield_tensorData.iloc[index, 8], mag_shield_tensorData.iloc[index, 9], mag_shield_tensorData.iloc[index, 10]]])
    return matrix

# Define file locations (for my local machine/in my working directory).
trainDataLocation = "C:\\Users\\lawre\\Downloads\\train.csv (1)\\train.csv"
structureDataLocation = "C:\\Users\\lawre\\Downloads\\train.csv (1)\\structures.csv"

# Load in data.
trainDataRaw = pd.read_csv(trainDataLocation, header=0)
structureDataRaw = pd.read_csv(structureDataLocation, header=0)

trainDataRaw = map_atom_info(trainDataRaw, 0)
trainDataRaw = map_atom_info(trainDataRaw, 1)

trainDataRaw['dist'] = trainDataRaw.apply(lambda x: dist(x), axis=1)
molecules=trainDataRaw.pop('molecule_name')

print (trainDataRaw)
#trainDataRaw['lengths'] = ""                                # Array for storing calculations.

# # Determine number of type/connection parameters for the molecules.
# resultTemp = trainDataRaw['type'].value_counts()
#
# # Determine list of unique molecule names.
# numAtoms = structureDataRaw['molecule_name'].value_counts()
# uniqueMolecules = structureDataRaw['molecule_name'].value_counts().index.values.tolist()
# uniqueMoleculeValues = []
# for a, b in zip(uniqueMolecules, numAtoms):
#     temp = int(a.split("_")[1])
#     uniqueMoleculeValues.append([temp,a,b])
#
# # Sort in ascending order.
# uniqueMoleculeValues.sort(key=lambda molecule: molecule[0])
#
# # Keep counter for index values for the following for loop.
# idCounter = 0
#
# for entry in uniqueMoleculeValues:
#
#     # TODO: Make separate entries; this will just put a list of bond distances in a single array per the first
#     # entry for each molecule.
#
#     # Create a list with the possible combinations.
#     combList = list(it.combinations(range(0,entry[2]),2))
#     combListInd = []
#     for x in combList:
#         if x[0]==0:
#             x=x[::-1]
#         combListInd.append(list(x))
#
#
#
#     # # Start calculating bond lengths.
#     # lengths = []
#     for i in combListInd:
#         trainDataInd=trainDataRaw.loc[(trainDataRaw['molecule_name']==structureDataRaw['molecule_name'][idCounter])&
#                             ((trainDataRaw['atom_index_0']==i[0]) | (trainDataRaw['atom_index_0']==i[1]))&
#                             ((trainDataRaw['atom_index_1']==i[1]) | (trainDataRaw['atom_index_1']==i[0]))].index
#
#         if len(trainDataInd)>0:
#             x1, y1, z1 = structureDataRaw.iloc[i[0]+idCounter,3], structureDataRaw.iloc[i[0]+idCounter,4], structureDataRaw.iloc[i[0]+idCounter,5]
#             x2, y2, z2 = structureDataRaw.iloc[i[1]+idCounter,3], structureDataRaw.iloc[i[1]+idCounter,4], structureDataRaw.iloc[i[1]+idCounter,5]
#             length = math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
#             trainDataRaw.loc[trainDataInd[0], 'lengths']=length
# #
# #     # Update index counter and store results in first index for each unique molecule.
# #     structureDataRaw.iat[idCounter, 6] = lengths
#     idCounter += entry[2]
# #
# # # Check to see if everything is a-OK.
# print(trainDataRaw.head(20))
# print (math.sqrt((structureDataRaw.iloc[18, 3]-structureDataRaw.iloc[17, 3])**2+(structureDataRaw.iloc[18, 4]-structureDataRaw.iloc[17, 4])**2+(structureDataRaw.iloc[18, 5]-structureDataRaw.iloc[17, 5])**2))