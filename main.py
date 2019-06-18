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

# Determine list of unique molecule names.
numAtoms = structureDataRaw['molecule_name'].value_counts()
uniqueMolecules = structureDataRaw['molecule_name'].value_counts().index.values.tolist()
uniqueMoleculeValues = []
for a, b in zip(uniqueMolecules, numAtoms):
    temp = int(a.split("_")[1])
    uniqueMoleculeValues.append([temp,a,b])

# Sort in ascending order.
uniqueMoleculeValues.sort(key=lambda molecule: molecule[0])

# Keep counter for index values for the following for loop.
idCounter = 0

for entry in uniqueMoleculeValues:

    # TODO: Make separate entries; this will just put a list of bond distances in a single array per the first
    # entry for each molecule.

    # Create a list with the possible combinations.
    combList = list(it.combinations(range(0,entry[2]),2))
    combListInd = []
    for x in combList:
        combListInd.append(list(x))

    # Start calculating bond lengths.
    lengths = []
    for i in combListInd:
        x1, y1, z1 = structureDataRaw.iloc[i[0],3], structureDataRaw.iloc[i[0],4], structureDataRaw.iloc[i[0],4]
        x2, y2, z2 = structureDataRaw.iloc[i[1],3], structureDataRaw.iloc[i[1],4], structureDataRaw.iloc[i[1],4]
        length = math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        lengths.append(length)

    # Update index counter and store results in first index for each unique molecule.
    structureDataRaw.iat[idCounter, 6] = lengths
    idCounter += entry[2]
    print(idCounter)

# Check to see if everything is a-OK.
print(structureDataRaw.head(10))

