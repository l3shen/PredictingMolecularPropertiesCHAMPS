import bondCalculator
import sklearn.model_selection as mls
import lightgbm as lgb
import os
import pandas as pd

# Define file locations (for my local machine/in my working directory).
trainDataLocation = "train.csv"
structureDataLocation = "structures.csv"
dipoleDataLocation = "dipole_moments.csv"

if os.path.isfile('trainDataPrepared.csv') == False:
    # Prepare data to include bond lengths.
    print("Processing data...")
    trainDataProc = bondCalculator.calcBondLengths(trainDataLocation, structureDataLocation)
elif os.path.isfile('trainDataPrepared.csv'):
    # Load in existing dataset.
    print("Loading data...")
    trainDataProc = pd.read_csv('trainDataPrepared.csv', header=0)

# Prepare train X and Y column names.
trainColumnsX = ['bond_dist']
trainColumnsY = ['scalar_coupling_constant']

# Perform K-fold split.
kfold = mls.KFold(n_splits=5, shuffle=True, random_state=0)
result = next(kfold.split(trainDataProc), None)
train = trainDataProc.iloc[result[0]]
test = trainDataProc.iloc[result[1]]

# Train model via lightGBM.
lgbTrain = lgb.Dataset(train[trainColumnsX], label=train[trainColumnsY])
lgbEval = lgb.Dataset(test[trainColumnsX], label=test[trainColumnsY])

# Model parameters.
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'mae'},
    'num_leaves': 25,
    'learning_rate': 0.0001,
    'num_iterations': 100000,
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
                early_stopping_rounds=5000)

print("Saving model.")
gbm.save_model('fitModel.txt')
print("Model saved.")