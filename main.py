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
import generateFeatures
import saveSubmissionFile
import importTestData
import sklearn.model_selection as mls
import lightgbm as lgb
import os
import pandas as pd
from collections import Counter

# Define file locations (for my local machine/in my working directory).
trainDataLocation = "C:\\Users\\lawre\\Downloads\\train.csv (1)\\train.csv"
testDataLocation = "C:\\Users\\lawre\\Downloads\\train.csv (1)\\test.csv"
structureDataLocation = "structureDataPrepared.csv" #NEW: created new structure csv by adding total number of atoms in each molecule using 'total_num_atoms.py'
#dipoleDataLocation = "dipole_moments.csv"

# Prepare train data.
if os.path.isfile('trainDataPrepared.csv') == False:
    # Prepare data to include bond lengths.
    print("Processing train data...")
    trainDataProc = bondCalculator.calcBondLengths(trainDataLocation, structureDataLocation, 'train')
elif os.path.isfile('trainDataPrepared.csv'):
    # Load in existing dataset.
    print("Loading train data...")
    trainDataProc = pd.read_csv('trainDataPrepared.csv', header=0)

#Prepare test data.
if os.path.isfile('testDataPrepared.csv') == False:
    # Prepare data to include bond lengths.
    print("Processing test data...")
    testDataProc = bondCalculator.calcBondLengths(testDataLocation, structureDataLocation, 'test')
elif os.path.isfile('testDataPrepared.csv'):
    # Load in existing dataset.
    print("Loading test data...")
    testDataProc = pd.read_csv('testDataPrepared.csv', header=0)


#NEW: creating new groups to groupby and generating new features
trainDataProc=generateFeatures.newGroupBy(trainDataProc)

trainDataProc, added_features0=generateFeatures.generate_features(trainDataProc, 'molecule_name')
trainDataProc, added_features1=generateFeatures.generate_features(trainDataProc, 'total_num_atoms')
trainDataProc, added_features2=generateFeatures.generate_features(trainDataProc, 'min_bond_dist_binned_x')
trainDataProc, added_features3=generateFeatures.generate_features(trainDataProc, 'atom_x')

added_features=added_features0 + added_features1 + added_features2 + added_features3
print (added_features)

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
trainColumnsX = ['bond_dist']+added_features
trainColumnsY = ['scalar_coupling_constant']

# Train params; set as same for all 8 data sets.
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'mae'},
    'num_leaves': 25,
    'learning_rate': 0.0001,
    'num_iterations': 200,
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
                    early_stopping_rounds=200)

    print("Saving model for type ", i, end=".\n")
    modelName = 'modelType' + str(i)
    gbm.save_model(modelName)
    models.append(gbm)
    print("Model saved.")
    # for loop end

print(models)

#TODO: Apply model to imported test data.
#NEW: add new groups and features to test data
testDataProc=generateFeatures.newGroupBy(testDataProc)

testDataProc, added_features0=generateFeatures.generate_features(testDataProc, 'molecule_name')
testDataProc, added_features1=generateFeatures.generate_features(testDataProc, 'total_num_atoms')
testDataProc, added_features2=generateFeatures.generate_features(testDataProc, 'min_bond_dist_binned_x')
testDataProc, added_features3=generateFeatures.generate_features(testDataProc, 'atom_x')
print(testDataProc.head())

testSubmissionX = ['bond_dist']+added_features
prediction = gbm.predict(testDataProc[testSubmissionX])
testDataSubmission = testDataProc[['id']]
testDataSubmission['scalar_coupling_constant'] = prediction # Can be made better. Use iloc.
print(testDataSubmission.head())
testDataSubmission.to_csv('submissionCSV.csv', index_label=False)
