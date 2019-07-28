import pandas as pd

def calcBondLengths(testDataLocation, structureDataLocation, saveResults=True):

    # Load in data.
    testDataRaw = pd.read_csv(testDataLocation, header=0)
    structureDataRaw = pd.read_csv(structureDataLocation, header=0)

    print("Preparing test data.")
    # Calculate bond lengths from structures akin to train data.
    # Map atomic coordinates to dataset.
    testData = map_atom_info(testDataRaw, structureDataRaw, 0)
    testData = map_atom_info(testData, structureDataRaw, 1)

    print("Calculating test bond lengths.")
    testData['bond_dist'] = testData.apply(lambda x: dist(x), axis=1)
    # testData['id'] = testDataRaw['id']

    if saveResults:
        print("Saving refined test dataset to CSV file.")
        testData.to_csv("testDataPrepared.csv")

    return testData


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
