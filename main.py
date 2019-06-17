import pandas as pd
import numpy as np

# Define file locations (for my local machine/in my working directory).
trainDataLocation = "train.csv"

# Load in data.
trainDataRaw = pd.read_csv(trainDataLocation, header=0)

# Determine number of type/connection parameters for the molecules.
