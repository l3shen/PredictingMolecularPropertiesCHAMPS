# Automatically updates params.
import pandas as pd
import pickle

def loadResults(save=True):

    paramsDict = []

    filenames = [
        'randHPO_results_0.csv',
        'randHPO_results_1.csv',
        'randHPO_results_2.csv',
        'randHPO_results_3.csv',
        'randHPO_results_4.csv',
        'randHPO_results_5.csv',
        'randHPO_results_6.csv',
        'randHPO_results_7.csv'
    ]

    for file in filenames:
        # Load in filename
        ds = pd.read_csv(file, header=0)

        # Set key column to int.
        ds['rank_test_score'] = ds['rank_test_score'].astype(int)

        # Sort column 'rank_test_score' to descending.
        ds = ds.sort_values(by=['rank_test_score'])

        # Pull out params for top ranked dataset.
        param = ds['params'].iloc[0]

        # Append to our list.
        paramsDict.append(dict(param))

        # Delete ds to clear up memory.
        del ds

    if save:

        # Save this to a pickle model.
        with open('params.pkl', 'wb') as f:
            pickle.dump(paramsDict, f)

    # Return list of dictionary values.
    return paramsDict

