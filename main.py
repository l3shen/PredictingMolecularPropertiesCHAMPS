import bondCalculator

# Define file locations (for my local machine/in my working directory).
trainDataLocation = "train.csv"
structureDataLocation = "structures.csv"
dipoleDataLocation = "dipole_moments.csv"

# Prepare data to include bond lengths.
trainDataProc = bondCalculator.calcBondLengths(trainDataLocation, structureDataLocation)

