import pandas as pd
import gridparams
import randomizedparams
import numpy as np
import time
import lightgbm as lgb
import sklearn.model_selection as mls


def gridsearch(start_num, end_num):
    best_params=[]
    for i in range(start_num, end_num+1):
        #split data into training and test sets
        features = trainData[i].drop(columns=dropColumns)
        labels = np.array(trainData[i]['scalar_coupling_constant']).reshape((-1,))
        train_features, test_features, train_labels, test_labels = mls.train_test_split(features, labels, test_size=0.2,
                                                                                        random_state=50)
        #create hyperparameter grid
        grid = mls.GridSearchCV(mdl, gridParams[i], verbose=0, cv=5, n_jobs=1)

        #apply LGBMRegressor to all sets of hyperparameters in grid
        grid.fit(train_features, train_labels)

        #save results as dataframe
        results = pd.DataFrame(grid.cv_results_)
        results.to_csv(f'gridHPO_results_{i}.csv')

        best_params+=[grid.best_params_]

    return best_params


def randsearch(start_num, end_num, num_iter):
    best_params=[]
    for i in range(start_num, end_num+1):
        #split data into training and test sets
        features = trainData[i].drop(columns=dropColumns)
        labels = np.array(trainData[i]['scalar_coupling_constant']).reshape((-1,))
        train_features, test_features, train_labels, test_labels = mls.train_test_split(features, labels, test_size=0.2,
                                                                                        random_state=50)

        #select random sets of hyperparameters from a very large grid
        grid = mls.RandomizedSearchCV(mdl, randParams[i], verbose=0, n_iter=num_iter, cv=5, n_jobs=1)

        #apply LGBMRegressor to randomly picked sets of hyperparameters
        grid.fit(train_features, train_labels)

        #save results as dataframe
        results = pd.DataFrame(grid.cv_results_)
        results.to_csv(f'randHPO_results_{i}.csv')

        best_params+=[grid.best_params_]

    return best_params

#begin timing
start=time.time()

#load data
print ('Loading Data')
trainData1=pd.read_csv('trainDataSet_0_3JHC.csv', header=0)
trainData2=pd.read_csv('trainDataSet_1_2JHC.csv', header=0)
trainData3=pd.read_csv('trainDataSet_2_1JHC.csv', header=0)
trainData4=pd.read_csv('trainDataSet_3_3JHH.csv', header=0)
trainData5=pd.read_csv('trainDataSet_4_2JHH.csv', header=0)
trainData6=pd.read_csv('trainDataSet_5_3JHN.csv', header=0)
trainData7=pd.read_csv('trainDataSet_6_2JHN.csv', header=0)
trainData8=pd.read_csv('trainDataSet_7_1JHN.csv', header=0)

trainData=[trainData1, trainData2, trainData3, trainData4, trainData5, trainData6, trainData7, trainData8]

#choose which columns to not include in model
dropColumns=['Unnamed: 0', 'id', 'molecule_name', 'atom_index_0', 'atom_index_1',
       'type', 'scalar_coupling_constant', 'Unnamed: 0_x', 'atom_0', 'x_0',
       'y_0', 'z_0', 'Unnamed: 0_y', 'atom_1', 'x_1', 'y_1', 'z_1',
       'total_num_atoms', 'atom_x', 'min_bond_dist_x',
       'min_bond_dist_binned_x', 'atom_y', 'min_bond_dist_y',
       'min_bond_dist_binned_y']

#initial set of parameters
print ('Constructing Model')
params = {'boosting_type': 'gbdt',
          'objective': 'regression',
          'num_leaves': 20,
          'learning_rate': 0.05,
          'feature_fraction': 0.5,
          'bagging_fraction': 0.5,
          'reg_alpha': 5,
          'reg_lambda': 10,
          'subsample': 0.5,
          'metric' : 'l1'}

#construct gradient boosting model
mdl = lgb.LGBMRegressor(boosting_type= 'gbdt',
          objective = 'regression',
          silent = True,
          num_leaves = params['num_leaves'],
          learning_rate = params['learning_rate'],
          feature_fraction = params['feature_fraction'],
          bagging_fraction = params['bagging_fraction'],
          reg_alpha = params['reg_alpha'],
            reg_lambda= params['reg_lambda'],
            subsample= params['subsample'],
            metric=params['metric'])

#import parameters for gridsearch
gridParams=[gridparams.params1,
            gridparams.params2,
            gridparams.params3,
            gridparams.params4,
            gridparams.params5,
            gridparams.params6,
            gridparams.params7,
            gridparams.params8]

#import parameters for randomsearch
randParams=[randomizedparams.params1,
            randomizedparams.params2,
            randomizedparams.params3,
            randomizedparams.params4,
            randomizedparams.params5,
            randomizedparams.params6,
            randomizedparams.params7,
            randomizedparams.params8]

print('Begin HPO')
randomsearch=randsearch(2, 3, 4)
print (randomsearch)

end=time.time()
#print time in minutes
print ((end-start)/60)