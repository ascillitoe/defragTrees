# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import sys
sys.path.append('../')
sys.path.append('./baselines/')

import os
import numpy as np
import pandas as pd
from sklearn.externals import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from defragTrees import DefragModel
from Baselines import inTreeModel, NHarvestModel, DTreeModel, BTreeModel

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

def run(prefix, Kmax, restart, trial, modeltype='regression', treenum=100, maxitr=1000, tol=1e-6, njobs=1):
    
    # trial
    dirname = './result/result_%s_itr' % (prefix,)
    for t in range(trial):
        
        # data
        dirname2 = '%s/result_%02d' % (dirname, t)
        trfile = '%s/%s_train_%02d.csv' % (dirname2, prefix, t)
        tefile = '%s/%s_test_%02d.csv' % (dirname2, prefix, t)
        Ztr = pd.read_csv(trfile, delimiter=',', header=None).values # Training data
        Xtr = Ztr[:, :-1]
        ytr = Ztr[:, -1]
        Zte = pd.read_csv(tefile, delimiter=',', header=None).values # Test data
        Xte = Zte[:, :-1]
        yte = Zte[:, -1]
        
        # Build scikit-learn random fores
        if modeltype=='regression':
            forest = RandomForestRegressor(n_estimators=treenum) 
            forest.fit(Xtr,ytr)
            score = mean_squared_error(yte,forest.predict(Xte))
        elif modeltype == 'classification':        
            forest = RandomForestClassifier(n_estimators=treenum)
            forest.fit(Xtr,ytr)    
            score = accuracy_score(yte,forest.predict(Xte))

        print('RF Test Error = %f' % (1.0-score,))
        cover = 1.0
        coll = 1.0
        np.savetxt('%s/res_rf_%02d.csv' % (dirname2, t), np.array([score, cover, coll, -1]), delimiter=',')

        # parse sklearn tree ensembles into the array of (feature index, threshold)
        splitter = DefragModel.parseSLtrees(forest) 

        # Write sklearn random forests to file (in the same format as the .R ones)
        # inTrees, NHarvest, BATrees and DTree2 can then be performed on this RF without any modification to packages.
        save_SLtrees('%s/forest/' % (dirname2,),forest)
#
#        # build R random forest
#        if modeltype == 'regression':
#            os.system('Rscript ./baselines/buildRegForest.R %s %s %s %d 0' % (trfile, tefile, dirname2, treenum))
#        elif modeltype == 'classification':        
#            os.system('Rscript ./baselines/buildClfForest.R %s %s %s %d 0' % (trfile, tefile, dirname2, treenum))
#        splitter = DefragModel.parseRtrees('%s/forest/' % (dirname2,))
#        zfile = '%s/pred_test.csv' % (dirname2,)
#        zte = pd.read_csv(zfile, delimiter=' ', header=None).values[:, -1]
#        if modeltype == 'regression':
#            score = np.mean((yte - zte)**2)
#            cover = 1.0
#            coll = 1.0
#        elif modeltype == 'classification':
#            score = np.mean(yte != zte)
#            cover = 1.0
#            coll = 1.0
#        np.savetxt('%s/res_rf_%02d.csv' % (dirname2, t), np.array([score, cover, coll, -1]), delimiter=',')
#
        # defragTrees
        mdl = DefragModel(modeltype=modeltype, restart=restart, maxitr=maxitr, tol=tol, seed=restart*t, njobs=njobs)
        mdl.fit(Xtr, ytr, splitter, Kmax, fittype='FAB')
        joblib.dump(mdl, '%s/%s_defrag_%02d.mdl' % (dirname2, prefix, t), compress=9)
        score, cover, coll = mdl.evaluate(Xte, yte)
        np.savetxt('%s/res_defrag_%02d.csv' % (dirname2, t), np.array([score, cover, coll, len(mdl.rule_)]), delimiter=',')
        
        # inTrees
        mdl2 = inTreeModel(modeltype=modeltype)
        mdl2.fit(Xtr, ytr, '%s/inTrees.txt' % (dirname2,))
        joblib.dump(mdl2, '%s/%s_inTrees_%02d.mdl' % (dirname2, prefix, t), compress=9)
        score, cover, coll = mdl2.evaluate(Xte, yte)
        np.savetxt('%s/res_inTrees_%02d.csv' % (dirname2, t), np.array([score, cover, coll, len(mdl2.rule_)]), delimiter=',')
        
        # NHarvest
        mdl3 = NHarvestModel(modeltype=modeltype)
        mdl3.fit(Xtr, ytr, '%s/nodeHarvest.txt' % (dirname2,))
        joblib.dump(mdl3, '%s/%s_nodeHarvest_%02d.mdl' % (dirname2, prefix, t), compress=9)
        score, cover, coll = mdl3.evaluate(Xte, yte)
        zfile = '%s/pred_test_nh.csv' % (dirname2,)
        zte = pd.read_csv(zfile, delimiter=' ', header=None).values[:, -1]
        if modeltype == 'regression':
            score = np.mean((yte - zte)**2)
        elif modeltype == 'classification':
            score = np.mean(yte != zte)
        np.savetxt('%s/res_nodeHarvest_%02d.csv' % (dirname2, t), np.array([score, cover, coll, len(mdl3.rule_)]), delimiter=',')
        
        # BA Tree Result
        mdl4 = BTreeModel(modeltype=modeltype, njobs=njobs, seed=t, verbose=True)
        mdl4.fit(Xtr, ytr, '%s/forest/' % (dirname2,))
        joblib.dump(mdl4, '%s/%s_BATree_%02d.mdl' % (dirname2, prefix, t), compress=9)
        score, cover, coll = mdl4.evaluate(Xte, yte)
        print('BATree Test Error = %f\n' % (score,))
        np.savetxt('%s/res_BATree_%02d.csv' % (dirname2, t), np.array([score, cover, coll, len(mdl4.rule_)]), delimiter=',')
        
        # DTree - depth = 2
        mdl5 = DTreeModel(modeltype=modeltype, max_depth=[2])
        mdl5.fit(Xtr, ytr)
        joblib.dump(mdl5, '%s/%s_DTree2_%02d.mdl' % (dirname2, prefix, t), compress=9)
        score, cover, coll = mdl5.evaluate(Xte, yte)
        np.savetxt('%s/res_DTree2_%02d.csv' % (dirname2, t), np.array([score, cover, coll, len(mdl5.rule_)]), delimiter=',')
    
    # summary
    plot_summarize(prefix, trial)
    summary2csv(prefix, trial)

def save_SLtrees(savedir,mdl):
    for t, tree in enumerate(mdl.estimators_): #loop through each tree in forest
        left = tree.tree_.children_left
        right = tree.tree_.children_right
        feature = tree.tree_.feature[left >= 0]
        threshold = tree.tree_.threshold[left >= 0]
        predictions = tree.tree_.value.flatten()
    
        n_nodes = tree.tree_.node_count

        os.makedirs(savedir, exist_ok=True)
        filename = os.path.join(savedir,'tree%03d.txt' % (t+1))
        f = open(filename, "w")       
        f.write(' node\t\t\t\t\t left\t\t\t right\t\t split var\t\t split point\t\t status\t\t prediction')

        s = 0 # counter for split nodes
        for n in range(n_nodes): # n is global node counter
            if(left[n] != right[n]): #if split node
                status = -3
                l = left[n]
                r = right[n]
                splitvar = feature[s] 
                splitpt = threshold[s]
                pred = predictions[n]
                s += 1
            else:  # leaf node
                status = -1
                l = 0
                r = 0
                splitvar = 0
                splitpt = 0.0
                pred = predictions[n]
            
            f.write('\n {:5d} \t {:9d} \t {:9d} \t {:9d} \t {:12.3f} \t {:9d} \t {:12.6f}'.format(n+1, l+1, r+1, splitvar+1, splitpt, status, pred ))

        f.close()
            

def summarize(prefix, trial):
    dirname = './result/result_%s_itr' % (prefix,)
    res = []
    for t in range(trial):
        dirname2 = '%s/result_%02d' % (dirname, t)
        res_t = []
        res_t.append(pd.read_csv('%s/res_rf_%02d.csv' % (dirname2, t), delimiter=',', header=None).values[:, 0])
        res_t.append(pd.read_csv('%s/res_defrag_%02d.csv' % (dirname2, t), delimiter=',', header=None).values[:, 0])
        res_t.append(pd.read_csv('%s/res_inTrees_%02d.csv' % (dirname2, t), delimiter=',', header=None).values[:, 0])
        res_t.append(pd.read_csv('%s/res_nodeHarvest_%02d.csv' % (dirname2, t), delimiter=',', header=None).values[:, 0])
        res_t.append(pd.read_csv('%s/res_BATree_%02d.csv' % (dirname2, t), delimiter=',', header=None).values[:, 0])
        res_t.append(pd.read_csv('%s/res_DTree2_%02d.csv' % (dirname2, t), delimiter=',', header=None).values[:, 0])
        res_t = np.asarray(res_t)
        res.append(res_t)
    res = np.array(res)
    return res
    
def print_summarize(prefix, trial):
    res = summarize(prefix, trial)
    res_s = np.std(res, axis=0)
    res = np.mean(res, axis=0)
    name = ['RandomForest', 'defragTrees', 'inTrees', 'NodeHarvest', 'BornAgainTree', 'DecisionTree2']
    for i in range(6):
        print('\t %s \t > TestError = %.3f (%.3f), Coverage = %.3f (%.3f), Overlap = %.3f (%.3f), Rules = %.3f (%.3f)' % (name[i], res[i, 0], res_s[i, 0], res[i, 1], res_s[i, 1], res[i, 2], res_s[i, 2], res[i, 3], res_s[i, 3]))

def plot_summarize(prefix, trial, title=None):
    res = summarize(prefix, trial)
    err = res[:, 1:, 0]
    num = res[:, 1:, 3]
    marker = ('bo', 'r^', 'gs', 'mp', 'c*')
    for i in range(err.shape[1]):
        plt.semilogx(num[:, i], err[:, i], marker[i], ms=12)
    plt.legend(['Proposed', 'inTrees', 'NH', 'BATree', 'DTree2'], numpoints=1)
    plt.xlabel('# of Rules', fontsize=20)
    plt.ylabel('Test Error', fontsize=20)
    if title is None:
        title = prefix
    plt.title(title, fontsize=24)
    plt.show()
    if not os.path.exists('./result/fig/'):
        os.mkdir('./result/fig/')
    plt.savefig('./result/fig/compare_%s.pdf' % (prefix,), format="pdf", bbox_inches="tight")
    plt.close()

def summary2csv(prefix, trial):
    res = summarize(prefix, trial)
    rows = ['RF', 'defrag', 'inTrees', 'NH', 'BATree', 'DTree2']
    r = res[:, :, [3, 0]].reshape((10, 12))
    idx = np.arange(6)
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
