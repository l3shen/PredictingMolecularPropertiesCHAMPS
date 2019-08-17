import saveSubmissionFile
import importTestData
import sklearn.model_selection as mls
import lightgbm as lgb
import os
import pandas as pd
from collections import Counter

def generate_features(DataProc, group):
    added_features=[]

    DataProc[f'{group}_atom_index_0_dist_min'] = DataProc.groupby([group, 'atom_index_0'])[
        'bond_dist'].transform('min')
    added_features+=[f'{group}_atom_index_0_dist_min']

    DataProc[f'{group}_atom_index_0_dist_max'] = DataProc.groupby([group, 'atom_index_0'])[
        'bond_dist'].transform('max')
    added_features += [f'{group}_atom_index_0_dist_max']

    DataProc[f'{group}_atom_index_1_dist_min'] = DataProc.groupby([group, 'atom_index_1'])[
        'bond_dist'].transform('min')
    added_features += [f'{group}_atom_index_1_dist_min']

    DataProc[f'{group}_atom_index_0_dist_mean'] = DataProc.groupby([group, 'atom_index_0'])[
        'bond_dist'].transform('mean')
    added_features += [f'{group}_atom_index_0_dist_mean']

    DataProc[f'{group}_atom_index_0_dist_std'] = DataProc.groupby([group, 'atom_index_0'])[
        'bond_dist'].transform('std')
    added_features += [f'{group}_atom_index_0_dist_std']

    DataProc[f'{group}_atom_index_1_dist_std'] = DataProc.groupby([group, 'atom_index_1'])[
        'bond_dist'].transform('std')
    added_features += [f'{group}_atom_index_1_dist_std']

    DataProc[f'{group}_atom_index_1_dist_max'] = DataProc.groupby([group, 'atom_index_1'])[
        'bond_dist'].transform('max')
    added_features += [f'{group}_atom_index_1_dist_max']

    DataProc[f'{group}_atom_index_1_dist_mean'] = DataProc.groupby([group, 'atom_index_1'])[
        'bond_dist'].transform('mean')
    added_features += [f'{group}_atom_index_1_dist_mean']

    DataProc[f'{group}_atom_index_0_dist_max_diff'] = DataProc[f'{group}_atom_index_0_dist_max'] - \
                                                           DataProc['bond_dist']
    added_features += [f'{group}_atom_index_0_dist_max_diff']

    DataProc[f'{group}_atom_index_0_dist_max_div'] = DataProc[f'{group}_atom_index_0_dist_max'] / \
                                                          DataProc['bond_dist']
    added_features += [f'{group}_atom_index_0_dist_max_div']

    DataProc[f'{group}_atom_index_0_dist_std_diff'] = DataProc[f'{group}_atom_index_0_dist_std'] - \
                                                           DataProc['bond_dist']
    added_features += [f'{group}_atom_index_0_dist_std_diff']

    DataProc[f'{group}_atom_index_0_dist_std_div'] = DataProc[f'{group}_atom_index_0_dist_std'] / \
                                                          DataProc['bond_dist']
    added_features += [f'{group}_atom_index_0_dist_std_div']

    DataProc[f'{group}_atom_0_couples_count'] = DataProc.groupby([group, 'atom_index_0'])['id'].transform(
        'count')
    added_features += [f'{group}_atom_0_couples_count']

    DataProc[f'{group}_atom_index_0_dist_min_div'] = DataProc[f'{group}_atom_index_0_dist_min'] / \
                                                          DataProc['bond_dist']
    added_features += [f'{group}_atom_index_0_dist_min_div']

    DataProc[f'{group}_atom_index_1_dist_std_diff'] = DataProc[f'{group}_atom_index_1_dist_std'] - \
                                                           DataProc['bond_dist']
    added_features += [f'{group}_atom_index_1_dist_std_diff']

    DataProc[f'{group}_atom_index_0_dist_mean_div'] = DataProc[f'{group}_atom_index_0_dist_mean'] / \
                                                           DataProc['bond_dist']
    added_features += [f'{group}_atom_index_0_dist_mean_div']

    DataProc[f'{group}_atom_1_couples_count'] = DataProc.groupby([group, 'atom_index_1'])['id'].transform(
        'count')
    added_features += [f'{group}_atom_1_couples_count']

    return DataProc, added_features

def newGroupBy(DataProc): #create new columns that is the distance of the nearest neighbor to an atom in the molecule as well as the identity of the atom (C, H, N, O)
    DataProc0=pd.DataFrame(DataProc[['molecule_name', 'atom_index_0', 'atom_1', 'bond_dist']]).rename(columns={'atom_index_0':'atom_index',
                                                                                                                         'atom_1': 'atom'})
    DataProc1=pd.DataFrame(DataProc[['molecule_name', 'atom_index_1', 'atom_0', 'bond_dist']]).rename(columns={'atom_index_1':'atom_index',
                                                                                                                         'atom_0': 'atom'})
    DataProc_concat=pd.concat([DataProc0, DataProc1])

    minDistances=DataProc_concat.sort_values('bond_dist').groupby(['molecule_name', 'atom_index'], as_index=False).first().rename(columns={'bond_dist': 'min_bond_dist'})
    minDistances['min_bond_dist_binned']=pd.cut(minDistances['min_bond_dist'], 20) #bin the values for grouping purposes

    DataProc=map_atom_info(DataProc, minDistances, 0)
    DataProc=map_atom_info(DataProc, minDistances, 1)

    return DataProc

def map_atom_info(df, structureData, atom_idx):
    df = pd.merge(df, structureData, how='left',
                  left_on=['molecule_name', f'atom_index_{atom_idx}'],
                  right_on=['molecule_name', 'atom_index'])

    df = df.drop('atom_index', axis=1)
    df = df.rename({'min_bond_dist': f'min_dist_{atom_idx}',
                    'atom': f'min_attached_{atom_idx}'})

    return df