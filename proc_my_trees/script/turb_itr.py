# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import sys
import os
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd
import sub_itr
from sklearn.model_selection import StratifiedKFold, train_test_split

# setting
prefix = 'model3'
seed = 0
kfold = 1
target = 'Stress anisotropy'

# data
X_data = pd.read_csv('./data/model3_Xdat.csv', sep=',', header=0)
Y_data = pd.read_csv('./data/model3_Ydat.csv', sep=',', header=0)
X_data = X_data.drop(columns='group')
Y_data = Y_data[target]
print(X_data.columns)

nrows = len(X_data.index) 
print('Total number of rows = ', nrows)

# data
if not os.path.exists('./result/'):
    os.mkdir('./result/')
dirname = './result/result_%s_itr' % (prefix,)
if not os.path.exists(dirname):
    os.mkdir(dirname)

if (kfold==1):
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, train_size=0.7, random_state=42)
    # save
    dirname2 = '%s/result_%02d' % (dirname, 0)
    if not os.path.exists(dirname2):
        os.mkdir(dirname2)
    df1 = pd.concat([X_train, Y_train], axis = 1)
    df2 = pd.concat([X_test , Y_test ], axis = 1)
    trfile = '%s/%s_train_%02d.csv' % (dirname2, prefix, 0)
    tefile = '%s/%s_test_%02d.csv' % (dirname2, prefix, 0)
    df1.to_csv(trfile, header=None, index=False)
    df2.to_csv(tefile, header=None, index=False)

else:
    k_fold = StratifiedKFold(n_splits=kfold, random_state=42,shuffle=True)
    cv = k_fold.split(X_data,Y_data)
    
    t = 0
    for train_index, test_index in cv:
        print('KFold = ', t)
        X_train, X_test = X_data.iloc[train_index], X_data.iloc[test_index]
        Y_train, Y_test = Y_data.iloc[train_index], Y_data.iloc[test_index]
    
        # save
        dirname2 = '%s/result_%02d' % (dirname, t)
        if not os.path.exists(dirname2):
            os.mkdir(dirname2)
        df1 = pd.concat([X_train, Y_train], axis = 1)
        df2 = pd.concat([X_test , Y_test ], axis = 1)
        trfile = '%s/%s_train_%02d.csv' % (dirname2, prefix, t)
        tefile = '%s/%s_test_%02d.csv' % (dirname2, prefix, t)
        df1.to_csv(trfile, header=None, index=False)
        df2.to_csv(tefile, header=None, index=False)
        t += 1

# demo_R
Kmax = 10
restart = 200
njobs = 4
treenum = 80
depth = 5
smear = 50
rftype='R'
sub_itr.run(prefix, Kmax, restart, kfold, treenum=treenum, depth=depth, modeltype='classification', njobs=njobs, smear_num=smear, rftype=rftype, scoring='f1')


