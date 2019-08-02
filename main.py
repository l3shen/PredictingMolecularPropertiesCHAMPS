import bondCalculator
import sklearn.model_selection as mls
import sklearn.preprocessing as prep
import lightgbm as lgb
import os
import pandas as pd

# Define file locations (for my local machine/in my working directory).
trainDataLocation = "C:\\Users\\lawre\\Downloads\\train.csv (1)\\train.csv"
structureDataLocation = "C:\\Users\\lawre\\Downloads\\train.csv (1)\\structures.csv"
predDataLocation = "C:\\Users\\lawre\\Downloads\\train.csv (1)\\test.csv"
dipoleDataLocation = "dipole_moments.csv"

if os.path.isfile('trainDataPrepared.csv') == False:
    # Prepare data to include bond lengths.
    print("Processing data...")
    trainDataProc = bondCalculator.calcBondLengths(trainDataLocation, structureDataLocation)
elif os.path.isfile('trainDataPrepared.csv'):
    # Load in existing dataset.
    print("Loading data...")
    trainDataProc = pd.read_csv('trainDataPrepared.csv', header=0)

le=prep.LabelEncoder()
ohe=prep.OneHotEncoder()

num_columns=trainDataProc.shape[1]

for i in range(0, num_columns):
    column_name=trainDataProc.columns[i]
    column_type=trainDataProc[column_name].dtypes
    if column_type == 'object':
        le.fit(trainDataProc[column_name])
        encoded_feature=le.transform(trainDataProc[column_name])
        trainDataProc[column_name]=pd.DataFrame(encoded_feature)

print (trainDataProc['type'])

# Prepare train X and Y column names.
trainColumnsX = ['type', 'bond_dist']
trainColumnsY = ['scalar_coupling_constant']

# Perform K-fold split.
kfold = mls.KFold(n_splits=5, shuffle=True, random_state=0)
result = next(kfold.split(trainDataProc), None)
train = trainDataProc.iloc[result[0]]
test = trainDataProc.iloc[result[1]]

# Train model via lightGBM.
lgbTrain = lgb.Dataset(train[trainColumnsX], label=train[trainColumnsY], categorical_feature=['type'])
lgbEval = lgb.Dataset(test[trainColumnsX], label=test[trainColumnsY])

# Model parameters.
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'mae'},
    'num_leaves': 25,
    'learning_rate': 0.0001,
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
                early_stopping_rounds=50)

print("Saving model.")
gbm.save_model('fitModel.txt')
print("Model saved.")
