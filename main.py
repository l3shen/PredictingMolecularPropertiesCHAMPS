import pandas as pd
import numpy as np
import itertools as it
import math

# Define file locations (for my local machine/in my working directory).
trainDataLocation = "train.csv"
structureDataLocation = "structures.csv"

# Load in data.
trainDataRaw = pd.read_csv(trainDataLocation, header=0)
structureDataRaw = pd.read_csv(structureDataLocation, header=0)
structureDataRaw['lengths'] = ""                                # Array for storing calculations.

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

# Keep counter for index values.
idCounter = 0

for entry in uniqueMoleculeValues:

    # Create a list with the possible combinations.
    combList = list(it.combinations(range(0,entry[2]),2))
    combListInd = [x + idCounter for x in combList]                # Keeping track by index.

    # Start calculating bond lengths.
    lengths = []
    for i in combListInd:
        x1, y1, y2 = structureDataRaw.loc[idCounter,i[3]], structureDataRaw.loc[idCounter,i[4]], structureDataRaw.loc[idCounter,i[5]]
        x1, y1, y2 = structureDataRaw.loc[idCounter, i[3]], structureDataRaw.loc[idCounter, i[4]], structureDataRaw.loc[
            idCounter, i[5]]
        length =




# for moleculePair in uniqueMoleculeValues:
#     if moleculePair[0] ==

# uniqueValues = (structureDataRaw['molecule_name'].value_counts().index.values.tolist(),
#                 structureDataRaw['molecule_name'].value_counts().values.tolist())
#
# print(uniqueValues)

# for molecule in uniqueValues:
#     if structureDataRaw['molecule_name'] == molecule:
