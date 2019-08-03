TYPES_LIST = [
    '3JHC',
    '2JHC',
    '1JHC',
    '3JHH',
    '2JHH',
    '3JHN',
    '2JHN',
    '1JHN'
]

import bondCalculator
import saveSubmissionFile
import importTestData
import sklearn.model_selection as mls
import lightgbm as lgb
import os
import pandas as pd

# Define file locations (for my local machine/in my working directory).
trainDataLocation = "train.csv"
testDataLocation = "test.csv"
structureDataLocation = "structures.csv"
dipoleDataLocation = "dipole_moments.csv"

# Prepare train data.
if os.path.isfile('trainDataPrepared.csv') == False:
    # Prepare data to include bond lengths.
    print("Processing train data...")
    trainDataProc = bondCalculator.calcBondLengths(trainDataLocation, structureDataLocation, 'train')
elif os.path.isfile('trainDataPrepared.csv'):
    # Load in existing dataset.
    print("Loading train data...")
    trainDataProc = pd.read_csv('trainDataPrepared.csv', header=0)

# Prepare test data.
if os.path.isfile('testDataPrepared.csv') == False:
    # Prepare data to include bond lengths.
    print("Processing test data...")
    testDataProc = bondCalculator.calcBondLengths(testDataLocation, structureDataLocation, 'test')
elif os.path.isfile('testDataPrepared.csv'):
    # Load in existing dataset.
    print("Loading test data...")
    testDataProc = pd.read_csv('testDataPrepared.csv', header=0)

# TODO: Split in to 8 data sets (x2 for train and test).
# Split data set in to 8.
typesDict = dict(tuple(trainDataProc.groupby('type')))

trainData1 = typesDict[TYPES_LIST[0]]
trainData2 = typesDict[TYPES_LIST[1]]
trainData3 = typesDict[TYPES_LIST[2]]
trainData4 = typesDict[TYPES_LIST[3]]
trainData5 = typesDict[TYPES_LIST[4]]
trainData6 = typesDict[TYPES_LIST[5]]
trainData7 = typesDict[TYPES_LIST[6]]
trainData8 = typesDict[TYPES_LIST[7]]

# Create a 'list' of these to simplify our for loop.
trainDataLoop = [trainData1,
                 trainData2,
                 trainData3,
                 trainData4,
                 trainData5,
                 trainData6,
                 trainData7,
                 trainData8
                 ]


# Prepare train X and Y column names.
trainColumnsX = ['bond_dist']
trainColumnsY = ['scalar_coupling_constant']

# Train params; set as same for all 8 data sets.
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'mae'},
    'num_leaves': 25,
    'learning_rate': 0.0001,
    'num_iterations': 500,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# Empty list of models for our training later.
models = []

# for loop start for training
for i in range(len(trainDataLoop)):

    # Perform K-fold split and prepare model.
    kfold = mls.KFold(n_splits=5, shuffle=True, random_state=0)
    result = next(kfold.split(trainDataLoop[i]), None)
    train = trainDataLoop[i].iloc[result[0]]
    test = trainDataLoop[i].iloc[result[1]]

    # Train model via lightGBM.
    lgbTrain = lgb.Dataset(train[trainColumnsX], train[trainColumnsY])
    lgbEval = lgb.Dataset(test[trainColumnsX], test[trainColumnsY])

    # Set up training.
    print("Beginning training for type ", i, end=".\n")
    gbm = lgb.train(params,
                    lgbTrain,
                    num_boost_round=200,
                    valid_sets=lgbEval,
                    early_stopping_rounds=500)

    print("Saving model for type ", i, end=".\n")
    modelName = 'modelType' + str(i)
    gbm.save_model(modelName)
    models.append(gbm)
    print("Model saved.")
    # for loop end

print(models)

# TODO: Apply model to imported test data.
# print(testDataProc.head())
# testSubmissionX = ['bond_dist']
# prediction = gbm.predict(testDataProc[testSubmissionX])
# testDataSubmission = testDataProc[['id']]
# testDataSubmission['scalar_coupling_constant'] = prediction # Can be made better. Use iloc.
# print(testDataSubmission.head())
# testDataSubmission.to_csv('submissionCSV.csv', index_label=False)
