import pandas as pd

def saveSubmissionFile(dataset, saveData=True):

    # Create a dataframe of only two columns akin to the sample submission file:
    columns = ['id', 'scalar_coupling_constant']

    # Extract key columns.
    submissionDataFrame = dataset[columns]

    # Save to CSV if required.
    if saveData:
        submissionDataFrame.to_csv("submissionDataSet.csv")

    # Return for funsies.
    return submissionDataFrame