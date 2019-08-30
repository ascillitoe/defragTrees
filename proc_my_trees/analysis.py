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
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import export_graphviz, DecisionTreeClassifier

def main():
    prefix = 'model3'
    trial = 1
    rftype = 'R'
    latex = True
    X_headers = ['Q-Criterion', 'Turbulence intensity', 'Turbulence Re',
       'Pgrad along streamline', 'turb/strain time-scale', 'Viscosity ratio',
       'Pressure/shear stresses', 'Vortex stretching',
       'Deviation from parallel shear', 'Convection/production of k',
       'total/normal stresses', 'CEV comparison']

    for t in range(trial):
        dirname = './result/result_%s_itr' % (prefix,)
        dirname2 = '%s/result_%02d' % (dirname, t)

        dirname3 = os.path.join(dirname,'rules')
        os.makedirs(dirname3,exist_ok=True) 

        dirname4 = os.path.join(dirname,'trees')
        os.makedirs(dirname4,exist_ok=True) 

        # defragTrees: write rules to file
        filename = '%s/%s_defrag_%02d.mdl' % (dirname2, prefix, t)
        mdl = joblib.load(filename)
        if latex:
            filename = '%s/%s_defrag_%02d.tex' % (dirname3, prefix, t)
            f = open(filename,'w')
            f.write( catchprint(mdl.printInLatex) )
            f.close()
        filename = '%s/%s_defrag_%02d.txt' % (dirname3, prefix, t)
        f = open(filename,'w')
        f.write(str(mdl))
        f.close()

        # inTrees: write rules to file
        filename = '%s/%s_inTrees_%02d.mdl' % (dirname2, prefix, t)
        mdl2 = joblib.load(filename)
        if latex:
            filename = '%s/%s_inTrees_%02d.tex' % (dirname3, prefix, t)
            f = open(filename,'w')
            f.write( catchprint(mdl2.printInLatex) )
            f.close()
        filename = '%s/%s_inTrees_%02d.txt' % (dirname3, prefix, t)
        f = open(filename,'w')
        f.write(str(mdl2))
        f.close()

        # nodeHarvest: write rules to file
        filename = '%s/%s_nodeHarvest_%02d.mdl' % (dirname2, prefix, t)
        mdl3 = joblib.load(filename)
        if latex:
            filename = '%s/%s_nodeHarvest_%02d.tex' % (dirname3, prefix, t)
            f = open(filename,'w')
            f.write( catchprint(mdl3.printInLatex) )
            f.close()
        filename = '%s/%s_nodeHarvest_%02d.txt' % (dirname3, prefix, t)
        f = open(filename,'w')
        f.write(str(mdl3))
        f.close()

        # BATree - visualise tree
        filename = '%s/%s_BATree_%02d.mdl' % (dirname2, prefix, t)
        mdl4 = joblib.load(filename)
        mytree = convert_BTree_to_DTree(mdl4.tree)
        export_DTree(mytree,'%s/%s_BATree_%02d.gv' % (dirname4, prefix, t),feature_names=X_headers)

        # DTree2 - visualise tree
        filename = '%s/%s_DTree2_%02d.mdl' % (dirname2, prefix, t)
        mdl5 = joblib.load(filename)
        export_DTree(mdl5.tree.tree_,'%s/%s_DTree2_%02d.gv' % (dirname4, prefix, t),feature_names=X_headers)

        # DTreeBA - visualise tree
        filename = '%s/%s_DTreeBA_%02d.mdl' % (dirname2, prefix, t)
        mdl6 = joblib.load(filename)
        export_DTree(mdl6.tree.tree_,'%s/%s_DTreeBA_%02d.gv' % (dirname4, prefix, t),feature_names=X_headers)

    # summary
    plot_summarize_std(prefix, trial, rftype)
    print_summarize(prefix, trial, rftype)
    summary2csv(prefix, trial, rftype)


def convert_BTree_to_DTree(tree):

    class Tree(object): pass  
    mytree = Tree()
    mytree.children_left  = tree.left_
    mytree.children_right = tree.right_
    mytree.threshold = tree.threshold_
    mytree.feature = tree.index_
    mytree.value = tree.pred_

    return mytree



def export_DTree(t,filename,feature_names=None):
    left      = t.children_left
    right     = t.children_right
    threshold = t.threshold  # leaf if this eq 0
    feature   = t.feature
    value     = np.squeeze(t.value)
    m = len(value)

    f = open('temp.txt','w')
    f.write(str(left))
    f.write('\n')
    f.write(str(right))
    parent = [-1] * m
    ctype = [-1] * m
    LI = [0]*m    # 0 for interior, 1 for leaf
    for i in range(m):
        if not left[i] == -1:
            parent[left[i]] = i
            ctype[left[i]] = 0
        if not right[i] == -1:
            parent[right[i]] = i
            ctype[right[i]] = 1
        if right[i] == -1 and left[i]==-1:
            LI[i] = 1

    f.write('\n')
    f.write(str(parent))
    f.write('\n')
    f.write(str(feature))
    f.write('\n')
    f.write(str(value))
    f.close()

    # Normalise value between 0 and 1 (for sklearn decision trees where value counts number of obs in either class)
    for i in range(m):
        val = value[i]
        value[i] = val/sum(val)

    # Open gv file and write header
    f= open(filename,"w")
    f.write('digraph tree { \n node [shape=box, style="rounded"] ; \n edge [fontname=helvetica] ;')

    # Write nodes
    for i in range(m):
        if(LI[i]==0): #Interior node
            if feature_names is None:
                f.write('\n%d [label="x[%d] <= %.3f"];' %(i, feature[i]+1,threshold[i]))
            else:
                f.write('\n%d [label="%s <= %.3f"];' %(i, feature_names[feature[i]],threshold[i]))
        else: #Leaf node
            P = value[i][1] # likelihood of y=1 (use as colorscale for green to red)
            H,S,V = getHSV(P)
            f.write( '\n' + r'%d [label="P(y=0):  %.2f\nP(y=1):  %.2f", style=filled color="black" fillcolor="%.4f %.4f %.4f"];' \
                    %(i,value[i][0],value[i][1],H,S,V) )

    # Node connectivity
    for i in range(m):
        if (parent[i]!=-1): # If not root node
            if (ctype[i]==0):
                TF = True
                LA = 45
            else:
                TF = False
                LA = -45
            f.write('\n%d->%d [labeldistance=2.5, labelangle=%d, headlabel=%s];' %(parent[i],i,LA,bool(TF)))

    # close gv file
    f.write('\n}')    
    f.close()

def getHSV(num):

    power = 1 - num
    H = power * 0.4 # Hue (note 0.4 = Green, see huge chart below)
    S = 0.9 # Saturation
    B = 0.9 # Brightness

    return [H,S,B]

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
        plt.plot(num[:, i], err[:, i], marker[i], ms=12)
    plt.plot([1,1e3],[np.mean(res[:,0,0]),np.mean(res[:,0,0])],'--k')
    if (rftype=='R'):
        plt.legend(['RF','defragTrees', 'inTrees', 'NH', 'BATree', 'DTree2'], numpoints=1)
    elif (rftype=='SL'):
        plt.legend(['RF','defragTrees', 'BATree', 'DTree2'], numpoints=1)
    plt.xlabel('# of Rules', fontsize=20)
    plt.ylabel('Test Error', fontsize=20)
    plt.xlim([1,2e2])
    plt.xscale('log')
    if title is None:
        title = prefix
    plt.title(title, fontsize=24)
    plt.show()

def plot_summarize_std(prefix, trial, rftype, title=None):
    res = summarize(prefix, trial, rftype)
    err = np.mean(res[:, 1:, 0],axis=0)
    num = np.mean(res[:, 1:, 3],axis=0)
    err_s = np.std(res[:,1:,0],axis=0)
    num_s = np.std(res[:,1:,3],axis=0)
    if (rftype=='R'):
        labels = ['defragTrees', 'inTrees', 'NH', 'BATree', 'DTree2']
    elif (rftype=='SL'):
        labels = ['defragTrees', 'BATree', 'DTree2']
    marker = ('o', '^', 's', 'p', '*')
    for i in range(len(err)):
        plt.errorbar(x=num[i], y=err[i], xerr=num_s[i], yerr=err_s[i], marker=marker[i], ms=10, capsize=5, capthick=2, lw=2,label=labels[i])
    plt.plot([1,1e3],[np.mean(res[:,0,0]),np.mean(res[:,0,0])],'--k')
    plt.xlabel('# of Rules', fontsize=20)
    plt.ylabel('Per-class test error', fontsize=20)
    plt.xlim([1,2e2])
    plt.xscale('log')
    if title is None:
        title = prefix
    plt.title(title, fontsize=24)

    # get handles
    handles, labels = plt.gca().get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    # use them in the legend
    plt.legend(handles, labels)

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

def save_tree(filename,mdl):
    tree = mdl
    print(dir(tree))

    left = tree.tree_.children_left
    right = tree.tree_.children_right
    feature = tree.tree_.feature[left >= 0]
    threshold = tree.tree_.threshold[left >= 0]
    predictions = tree.tree_.value.flatten()

    n_nodes = tree.tree_.node_count

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

def catchprint(func):
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        func()

    s = f.getvalue()

    return s

if __name__ == '__main__':
    main()
