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
    trainDataProc = bondCalculator.calcBondLengths(trainDataLocation, structureDataLocation)
elif os.path.isfile('trainDataPrepared.csv'):
    # Load in existing dataset.
    print("Loading train data...")
    trainDataProc = pd.read_csv('trainDataPrepared.csv', header=0)

# Prepare test data.
if os.path.isfile('testDataPrepared.csv') == False:
    # Prepare data to include bond lengths.
    print("Processing test data...")
    testDataProc = importTestData.calcBondLengths(testDataLocation, structureDataLocation)
elif os.path.isfile('testDataPrepared.csv'):
    # Load in existing dataset.
    print("Loading test data...")
    testDataProc = pd.read_csv('testDataPrepared.csv', header=0)

# Prepare train X and Y column names.
trainColumnsX = ['bond_dist']
trainColumnsY = ['scalar_coupling_constant']

# Perform K-fold split and prepare model.
kfold = mls.KFold(n_splits=5, shuffle=True, random_state=0)
result = next(kfold.split(trainDataProc), None)
train = trainDataProc.iloc[result[0]]
test = trainDataProc.iloc[result[1]]


# Train model via lightGBM.
lgbTrain = lgb.Dataset(train[trainColumnsX], train[trainColumnsY])
lgbEval = lgb.Dataset(test[trainColumnsX], test[trainColumnsY])

# Model parameters.
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

# Set up training.
print("Beginning training.")
gbm = lgb.train(params,
                lgbTrain,
                num_boost_round=200,
                valid_sets=lgbEval,
                early_stopping_rounds=500)

print("Saving model.")
gbm.save_model('fitModel.txt')
print("Model saved.")


# TODO: Apply model to imported test data.
print(testDataProc.head())
testSubmissionX = ['bond_dist']
prediction = gbm.predict(testDataProc[testSubmissionX])
testDataSubmission = testDataProc[['id']]
testDataSubmission['scalar_coupling_constant'] = prediction # Can be made better. Use iloc.
print(testDataSubmission.head())
testDataSubmission.to_csv('submissionCSV.csv', index_label=False)
