# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara (modified by Ashley Scillitoe)
"""

import sys
sys.path.append('../')
sys.path.append('./baselines/')

import os
import numpy as np
import pandas as pd
from sklearn.externals import joblib

import matplotlib.pyplot as plt

from defragTrees import DefragModel
from Baselines import inTreeModel, NHarvestModel, DTreeModel, BTreeModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import GridSearchCV

def run(prefix, Kmax, restart, trial, modeltype='regression', rftype='R', treenum=100, depth=20, maxitr=1000, tol=1e-6, njobs=1,smear_num=100,verbose=True,scoring='standard'):
    
    # trial
    dirname = './result/result_%s_itr' % (prefix,)
    for t in range(trial):
        if verbose:
            print('\n***********************************')
            print('Fold: %02d/%02d' %(t+1,trial))
            print(  '***********************************')

        # data
        dirname2 = '%s/result_%02d' % (dirname, t)
        trfile = '%s/%s_train_%02d.csv' % (dirname2, prefix, t)
        tefile = '%s/%s_test_%02d.csv' % (dirname2, prefix, t)
        Ztr = pd.read_csv(trfile, delimiter=',', header=None).values
        Xtr = Ztr[:, :-1]
        ytr = Ztr[:, -1]
        Zte = pd.read_csv(tefile, delimiter=',', header=None).values
        Xte = Zte[:, :-1]
        yte = Zte[:, -1]
       
        if (rftype=='R'):
            # build R random forest
            if modeltype == 'regression':
                os.system('Rscript ./baselines/buildRegForest.R %s %s %s %d 0' % (trfile, tefile, dirname2, treenum))
            elif modeltype == 'classification':        
                if verbose: print('Running buildClfForest.R')
                os.system('Rscript ./baselines/buildClfForest.R %s %s %s %d 0' % (trfile, tefile, dirname2, treenum))
            if verbose: print('Running DefragModel.parseRtrees')
            splitter = DefragModel.parseRtrees('%s/forest/' % (dirname2,))
            zfile = '%s/pred_test.csv' % (dirname2,)
            zte = pd.read_csv(zfile, delimiter=' ', header=None).values[:, -1]
            if modeltype == 'regression':
                score = np.mean((yte - zte)**2)
                cover = 1.0
                coll = 1.0
            elif modeltype == 'classification':
                if (scoring=='balanced'):
                    from sklearn.metrics import balanced_accuracy_score
                    score = 1.0 - balanced_accuracy_score(yte,zte)
                elif (scoring=='standard'):
                    from sklearn.metrics import accuracy_score
                    score = 1.0 - accuracy_score(yte,zte)
                elif (scoring=='f1'):
                    from sklearn.metrics import f1_score
                    score = f1_score(yte,zte)
                cover = 1.0
                coll = 1.0

        elif (rftype=='SL'):
            if modeltype=='regression':
                forest = RandomForestRegressor(n_estimators=treenum,n_jobs=njobs)
                forest.fit(Xtr,ytr)
                score = mean_squared_error(yte,forest.predict(Xte))
            elif modeltype == 'classification':
                forest = RandomForestClassifier(n_estimators=treenum,n_jobs=njobs, max_depth=depth)
                forest.fit(Xtr,ytr)
#                parameters = {'max_leaf_nodes':[16, 32, 64, 128], 'min_samples_leaf':[10,50,100,150]}
#                forest = RandomForestClassifier(n_estimators=treenum,n_jobs=njobs)
#                gs = GridSearchCV(forest, parameters, cv=3)
#                gs.fit(Xtr,ytr)
#                print(gs.best_params_)
#                forest = gs.best_estimator_
                if (scoring=='balanced'):
                    from sklearn.metrics import balanced_accuracy_score
                    score = 1.0 - balanced_accuracy_score(yte,zte)
                elif (scoring=='standard'):
                    from sklearn.metrics import accuracy_score
                    score = 1.0 - accuracy_score(yte,zte)
                elif (scoring=='f1'):
                    from sklearn.metrics import f1_score
                    score = f1_score(yte,zte)
            cover = 1.0
            coll = 1.0

            # parse sklearn tree ensembles into the array of (feature index, threshold)
            splitter = DefragModel.parseSLtrees(forest) 

            # Write sklearn random forests to file (in the same format as the .R ones)
            # inTrees, NHarvest, BATrees and DTree2 can then be performed on this RF without any modification to packages.
            DefragModel.save_SLtrees('%s/forest/' % (dirname2,),forest)
        
        print('RF Test Score = %.2f' % (score))
        print('RF Test Coverage = %.2f' % (cover))
        print('RF Overlap = %.2f' % (coll))
        np.savetxt('%s/res_rf_%02d.csv' % (dirname2, t), np.array([score, cover, coll, -1]), delimiter=',')

        # defragTrees
        if verbose: print('Defragging model')
        mdl = DefragModel(modeltype=modeltype, restart=restart, maxitr=maxitr, tol=tol, seed=restart*t, njobs=njobs,score=scoring)
        if verbose: print('Fitting defrag')
        mdl.fit(Xtr, ytr, splitter, Kmax, fittype='FAB')
        joblib.dump(mdl, '%s/%s_defrag_%02d.mdl' % (dirname2, prefix, t), compress=9)
        score, cover, coll = mdl.evaluate(Xte, yte)
        print('Defrag Test Score = %.2f' % (score))
        print('Defrag Test Coverage = %.2f' % (cover))
        print('Defrag Overlap = %.2f' % (coll))
        np.savetxt('%s/res_defrag_%02d.csv' % (dirname2, t), np.array([score, cover, coll, len(mdl.rule_)]), delimiter=',')

        if (rftype=='R'):
            # inTrees
            if verbose: print('Fitting inTree model')
            mdl2 = inTreeModel(modeltype=modeltype,score=scoring)
            mdl2.fit(Xtr, ytr, '%s/inTrees.txt' % (dirname2,))
            joblib.dump(mdl2, '%s/%s_inTrees_%02d.mdl' % (dirname2, prefix, t), compress=9)
            score, cover, coll = mdl2.evaluate(Xte, yte)
            print('inTrees Test Score = %.2f' % (score))
            print('inTrees Test Coverage = %.2f' % (cover))
            print('inTrees Overlap = %.2f' % (coll))
            np.savetxt('%s/res_inTrees_%02d.csv' % (dirname2, t), np.array([score, cover, coll, len(mdl2.rule_)]), delimiter=',')
            
            # NHarvest
            if verbose: print('Fitting NHarvest model')
            mdl3 = NHarvestModel(modeltype=modeltype,score=scoring)
            mdl3.fit(Xtr, ytr, '%s/nodeHarvest.txt' % (dirname2,))
            joblib.dump(mdl3, '%s/%s_nodeHarvest_%02d.mdl' % (dirname2, prefix, t), compress=9)
            score, cover, coll = mdl3.evaluate(Xte, yte)
            zfile = '%s/pred_test_nh.csv' % (dirname2,)
            zte = pd.read_csv(zfile, delimiter=' ', header=None).values[:, -1]
            if modeltype == 'regression':
                score = np.mean((yte - zte)**2)
            elif modeltype == 'classification':
                if (score=='balanced'):
                    from sklearn.metrics import balanced_accuracy_score
                    score = 1.0 - balanced_accuracy_score(yte,zte)
                elif (score=='standard'):
                    from sklearn.metrics import accuracy_score
                    score = 1.0 - accuracy_score(yte,zte)
                elif (score=='f1'):
                    from sklearn.metrics import f1_score
                    score = f1_score(yte,zte)
            print('NHarvest Test Score = %.2f' % (score))
            print('NHarvest Test Coverage = %.2f' % (cover))
            print('NHarvest Overlap = %.2f' % (coll))
            np.savetxt('%s/res_nodeHarvest_%02d.csv' % (dirname2, t), np.array([score, cover, coll, len(mdl3.rule_)]), delimiter=',')
        
        # BA Tree Result
        if verbose: print('Fitting BATree model')
        mdl4 = BTreeModel(modeltype=modeltype, njobs=njobs, seed=t, smear_num=50,score=scoring)
        mdl4.fit(Xtr, ytr, '%s/forest/' % (dirname2,))
        joblib.dump(mdl4, '%s/%s_BATree_%02d.mdl' % (dirname2, prefix, t), compress=9)
        score, cover, coll = mdl4.evaluate(Xte, yte)
        print('BATree Test Score = %.2f' % (score))
        print('BATree Test Coverage = %.2f' % (cover))
        print('BATree Overlap = %.2f' % (coll))
        np.savetxt('%s/res_BATree_%02d.csv' % (dirname2, t), np.array([score, cover, coll, len(mdl4.rule_)]), delimiter=',')
        
        # DTree - depth = 2
        if verbose: print('Fitting Dtree')
        mdl5 = DTreeModel(modeltype=modeltype, max_depth=[2],score=scoring)
        mdl5.fit(Xtr, ytr)
        joblib.dump(mdl5, '%s/%s_DTree2_%02d.mdl' % (dirname2, prefix, t), compress=9)
        score, cover, coll = mdl5.evaluate(Xte, yte)
        print('DTree Test Score = %.2f' % (score))
        print('DTree Test Coverage = %.2f' % (cover))
        print('DTree Overlap = %.2f' % (coll))
        np.savetxt('%s/res_DTree2_%02d.csv' % (dirname2, t), np.array([score, cover, coll, len(mdl5.rule_)]), delimiter=',')
    
    # summary
    plot_summarize(prefix, trial, rftype)
    summary2csv(prefix, trial, rftype)
    
def summarize(prefix, trial, rftype):
    dirname = './result/result_%s_itr' % (prefix,)
    res = []
    for t in range(trial):
        dirname2 = '%s/result_%02d' % (dirname, t)
        res_t = []
        res_t.append(pd.read_csv('%s/res_rf_%02d.csv' % (dirname2, t), delimiter=',', header=None).values[:, 0])
        res_t.append(pd.read_csv('%s/res_defrag_%02d.csv' % (dirname2, t), delimiter=',', header=None).values[:, 0])
        if (rftype=='R'):
            res_t.append(pd.read_csv('%s/res_inTrees_%02d.csv' % (dirname2, t), delimiter=',', header=None).values[:, 0])
            res_t.append(pd.read_csv('%s/res_nodeHarvest_%02d.csv' % (dirname2, t), delimiter=',', header=None).values[:, 0])
        res_t.append(pd.read_csv('%s/res_BATree_%02d.csv' % (dirname2, t), delimiter=',', header=None).values[:, 0])
        res_t.append(pd.read_csv('%s/res_DTree2_%02d.csv' % (dirname2, t), delimiter=',', header=None).values[:, 0])
        res_t = np.asarray(res_t)
        res.append(res_t)
    res = np.array(res)
    return res
    
def print_summarize(prefix, trial, rftype):
    res = summarize(prefix, trial, rftype)
    res_s = np.std(res, axis=0)
    res = np.mean(res, axis=0)
    if (rftype=='R'):
        name = ['RandomForest', 'defragTrees', 'inTrees', 'NodeHarvest', 'BornAgainTree', 'DecisionTree2']
    elif (rftype=='SL'):
        name = ['RandomForest', 'defragTrees', 'BornAgainTree', 'DecisionTree2']
    for i in range(6):
        print('\t %s \t > TestError = %.3f (%.3f), Coverage = %.3f (%.3f), Overlap = %.3f (%.3f), Rules = %.3f (%.3f)' % (name[i], res[i, 0], res_s[i, 0], res[i, 1], res_s[i, 1], res[i, 2], res_s[i, 2], res[i, 3], res_s[i, 3]))

def plot_summarize(prefix, trial, rftype, title=None):
    res = summarize(prefix, trial, rftype)
    err = res[:, 1:, 0]
    num = res[:, 1:, 3]
    marker = ('bo', 'r^', 'gs', 'mp', 'c*')
    for i in range(err.shape[1]):
        plt.semilogx(num[:, i], err[:, i], marker[i], ms=12)
    if (rftype=='R'):
        plt.legend(['Proposed', 'inTrees', 'NH', 'BATree', 'DTree2'], numpoints=1)
    elif (rftype=='SL'):
        plt.legend(['Proposed', 'BATree', 'DTree2'], numpoints=1)
    plt.xlabel('# of Rules', fontsize=20)
    plt.ylabel('Test Error', fontsize=20)
    if title is None:
        title = prefix
    plt.title(title, fontsize=24)
    plt.show()

def summary2csv(prefix, trial, rftype):
    res = summarize(prefix, trial, rftype)
    if (rftype=='R'):
        rows = ['RF', 'defrag', 'inTrees', 'NH', 'BATree', 'DTree2']
        r = res[:, :, [3, 0]].reshape((trial, 12))
        idx = np.arange(6)
    elif (rftype=='SL'):
        rows = ['RF', 'defrag', 'BATree', 'DTree2']
        r = res[:, :, [3, 0]].reshape((trial, 8))
        idx = np.arange(4)
    res = np.array([[0, 0, 'a']])
    for i in idx:
        res = np.r_[res, np.c_[r[:, (2 * i):(2 * i + 2)], np.array([rows[i]] * trial)]]
    res = np.r_[res, np.array([[1, np.mean(r[:, 1]), 'RF']])]
    res = np.r_[res, np.array([[1000, np.mean(r[:, 1]), 'RF']])]
    if not os.path.exists('./result/csv/'):
        os.mkdir('./result/csv/')
    df = pd.DataFrame(res[1:, :])
    df.columns = ['rule', 'error', 'label']
    df.to_csv('./result/csv/csv_%s.csv' % (prefix,), index=None)
