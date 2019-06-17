import pandas as pd
import numpy as np
import itertools as it

# Define file locations (for my local machine/in my working directory).
trainDataLocation = "train.csv"
structureDataLocation = "structures.csv"

# Load in data.
trainDataRaw = pd.read_csv(trainDataLocation, header=0)
structureDataRaw = pd.read_csv(structureDataLocation, header=0)

# Determine number of type/connection parameters for the molecules.
resultTemp = trainDataRaw['type'].value_counts()
print(resultTemp)

# print(list(it.combinations([1,2,3,4], 2)))

# Determine list of unique molecule names.
numAtoms = structureDataRaw['molecule_name'].value_counts()
uniqueMolecules = structureDataRaw['molecule_name'].value_counts().index.values.tolist()
uniqueMoleculeValues = []
for a, b in zip(uniqueMolecules, numAtoms):
    temp = int(a.split("_")[1])
    uniqueMoleculeValues.append([temp,a,b])

uniqueMoleculeValues.sort(key=lambda molecule: molecule[0])
print(uniqueMoleculeValues[0:5])

# for moleculePair in uniqueMoleculeValues:
#     if moleculePair[0] ==

# uniqueValues = (structureDataRaw['molecule_name'].value_counts().index.values.tolist(),
#                 structureDataRaw['molecule_name'].value_counts().values.tolist())
#
# print(uniqueValues)

# for molecule in uniqueValues:
#     if structureDataRaw['molecule_name'] == molecule:
