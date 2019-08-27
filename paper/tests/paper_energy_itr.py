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
import paper_sub_itr

# setting
prefix = 'energy'
seed = 0
m1 = 384
m2 = m1 + 384
trial = 1  # should be 10

# data
df = pd.read_csv('./data/energy.csv', sep=',', header=0)
df = df.drop(columns=['Y2','Unnamed: 10','Unnamed: 11'])
num = len(df)

# data
if not os.path.exists('./result/'):
    os.mkdir('./result/')
dirname = './result/result_%s_itr' % (prefix,)
if not os.path.exists(dirname):
    os.mkdir(dirname)
for t in range(trial):
    
    # data - train & test
    np.random.seed(seed + t)
    idx = np.random.permutation(num)
    df1 = df.iloc[idx[:m1], :]
    df2 = df.iloc[idx[m1:m2], :]
    
    # save
    dirname2 = '%s/result_%02d' % (dirname, t)
    if not os.path.exists(dirname2):
        os.mkdir(dirname2)
    trfile = '%s/%s_train_%02d.csv' % (dirname2, prefix, t)
    tefile = '%s/%s_test_%02d.csv' % (dirname2, prefix, t)
    df1.to_csv(trfile, header=None, index=False)
    df2.to_csv(tefile, header=None, index=False)

# demo_R
Kmax = 10
restart = 200
njobs = 4
treenum = 100
paper_sub_itr.run(prefix, Kmax, restart, trial, treenum=treenum, modeltype='regression', njobs=njobs)
