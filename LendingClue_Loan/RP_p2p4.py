#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 12:19:43 2018

@author: syan
"""

#%% Imports
import pandas as pd
import numpy as np
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict
from helpers import   pairwiseDistCorr,nn_reg,nn_arch,reconstructionError
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import SparseRandomProjection
from itertools import product
import matplotlib.pyplot as plt
out = './Part2/RP/'
cmap = cm.get_cmap('Spectral') 


np.random.seed(0)
loans = pd.read_csv('./lending_club_scaled.csv')
loans_X = loans.drop('safe_loans',1).copy().values
loans_Y = loans['safe_loans'].copy().values

blocks = pd.read_csv('./page-blocks.csv')
class1 = blocks[blocks['Class']==1]
class1 = class1.sample(frac = 0.1, random_state = 1)
blocks_balanced = class1
for n in range(2,6):
    blocks_balanced = blocks_balanced.append(blocks[blocks['Class']==n]) 
blocks_X = blocks_balanced.drop('Class',1).copy().values
blocks_Y = blocks_balanced['Class'].copy().values
blocks_X= StandardScaler().fit_transform(blocks_X)
print blocks_X.shape

# Run RP for Loans 10 times
#clusters =  [2,5,10,15,20,25,30,35,40]
dims = [2,4,6,9,12,15,18,21,26]

tmp_distcorr = defaultdict(dict)
tmp_recnstErr = defaultdict(dict)

for i,dim in product(range(10),dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(loans_X)    
    tmp_recnstErr[dim][i] = reconstructionError(rp, loans_X)
    tmp_distcorr[dim][i] = pairwiseDistCorr(rp.transform(loans_X), loans_X)
tmp_distcorr =pd.DataFrame(tmp_distcorr).T
tmp_distcorr['mean'] = np.mean(tmp_distcorr.iloc[:,0:10], axis = 1)
tmp_distcorr['std'] = np.std(tmp_distcorr.iloc[:,0:10], axis = 1)

tmp_recnstErr =pd.DataFrame(tmp_recnstErr).T
tmp_recnstErr['mean'] = np.mean(tmp_recnstErr.iloc[:,0:10], axis = 1)
tmp_recnstErr['std'] = np.std(tmp_recnstErr.iloc[:,0:10], axis =1)

tmp_distcorr.to_csv(out+'loans_RP_distCorr.csv')
tmp_recnstErr.to_csv(out+'loans_RP_reconstrErr.csv')


#%%
# Run RP for Pageblock 10 times
dims = [2,3,4,5,6,7,8,9,10]
tmp_distcorr = defaultdict(dict)
tmp_recnstErr = defaultdict(dict)

for i,dim in product(range(10),dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(blocks_X)    
    tmp_recnstErr[dim][i] = reconstructionError(rp, blocks_X)
    tmp_distcorr[dim][i] = pairwiseDistCorr(rp.transform(blocks_X), blocks_X)
tmp_distcorr =pd.DataFrame(tmp_distcorr).T
tmp_distcorr['mean'] = np.mean(tmp_distcorr.iloc[:,0:10], axis = 1)
tmp_distcorr['std'] = np.std(tmp_distcorr.iloc[:,0:10], axis = 1)

tmp_recnstErr =pd.DataFrame(tmp_recnstErr).T
tmp_recnstErr['mean'] = np.mean(tmp_recnstErr.iloc[:,0:10], axis = 1)
tmp_recnstErr['std'] = np.std(tmp_recnstErr.iloc[:,0:10], axis =1)
tmp_distcorr.to_csv(out+'pb_RP_distCorr.csv')
tmp_recnstErr.to_csv(out+'pb_RP_reconstrErr.csv')



#%% Validation for part 2
dims1 = [2,4,6,9,12,15,18,21,26]

grid ={'rp__n_components':dims1}
rp = SparseRandomProjection(random_state=5)       
mlp = MLPClassifier(solver='lbfgs', activation='identity', alpha=0.1,
                     hidden_layer_sizes=(50,),
                    max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('rp',rp),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(loans_X, loans_Y)
tmp1 = pd.DataFrame(gs.cv_results_)
tmp1.to_csv(out+'Loans_RP_dim_red.csv')

dims2 = [2,3,4,5,6,7,8,9,10]
grid ={'rp__n_components':dims2}
rp = SparseRandomProjection(random_state=5)           
mlp = MLPClassifier(solver='lbfgs', activation='logistic', alpha=0.1,
                     hidden_layer_sizes=(50,),
                    max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('rp',rp),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(blocks_X,blocks_Y)
tmp2 = pd.DataFrame(gs.cv_results_)
tmp2.to_csv(out+'Pageblock_RP_dim_red.csv')

#%% For part 4
out = './Part4/'
rp = SparseRandomProjection(random_state=5, n_components=4)  
blocks_X2 = rp.fit_transform(blocks_X)
grid ={'NN__alpha':nn_reg}   
mlp = MLPClassifier(solver='lbfgs', activation='logistic', 
                     hidden_layer_sizes=(50,),
                    max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

start = time()
gs.fit(blocks_X2,blocks_Y)
print "Benchmark run time: %.2f seconds" %(time() - start)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Pageblocks_RP_NN.csv')