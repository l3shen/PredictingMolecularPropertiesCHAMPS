import pandas as pd
import numpy as np
import itertools as it
import math
import time
import sys


structureDataLocation = "C:\\Users\\lawre\\Downloads\\train.csv (1)\\structures.csv"

structuredata=pd.read_csv(structureDataLocation, header=0)


structuredata['total_num_atoms']=structuredata.groupby(['molecule_name'])['atom'].transform('count')

structuredata.to_csv('structureDataPrepared.csv')



