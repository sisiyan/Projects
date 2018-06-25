#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 11:05:42 2018

@author: syan
"""

#%% Imports
import pandas as pd
import numpy as np
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import nn_arch, nn_reg
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt


out = './Part2/ICA/'

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




#%% Plot for part2
ica = FastICA(random_state=5)
kurt1 = {}
for dim in range(1, 27):
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(loans_X)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt1[dim] = tmp.abs().mean()

kurt1 = pd.Series(kurt1) 
kurt1.to_csv(out+'ICA_loans_screen.csv')


ica = FastICA(random_state=5)
kurt2 = {}
for dim in range(1,11):
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(blocks_X)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt2[dim] = tmp.abs().mean()
    print tmp

kurt2 = pd.Series(kurt2) 
kurt2.to_csv(out+'ICA_pb_screen.csv')

ax = kurt1.plot(title = "Loans ICA Kurtosis",\
                style = '.')
ax.set_xlabel("Dimension")
ax.set_ylabel("Kurtosis")  
plt.savefig(out+'Loans_part2_ICA_Kurt.png', dpi=300)
plt.close()


ax = kurt2.plot(title = "Pageblocks ICA Kurtosis",\
                style = '.')
ax.set_xlabel("Dimension")
ax.set_ylabel("Kurtosis")  
plt.savefig(out+'pageblocks_part2_ICA_Kurt.png', dpi=300)
plt.close()

#%% Validation for part 2
dims1 = [5,10,15,20, 21, 22, 23, 26]

grid ={'ica__n_components':dims1}
ica = FastICA(random_state=5)       
mlp = MLPClassifier(solver='lbfgs', activation='identity', alpha=0.1,
                     hidden_layer_sizes=(50,),
                    max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('ica',ica),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(loans_X, loans_Y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'loans_ica_dim_red.csv')

dims2 = [2,3,4,5,6, 7, 8, 9, 10]
grid ={'ica__n_components':dims2}
ica = FastICA(random_state=5)       
mlp = MLPClassifier(solver='lbfgs', activation='logistic', alpha=0.1,
                     hidden_layer_sizes=(50,),
                    max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('ica',ica),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(blocks_X, blocks_Y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'pageblock_ica_dim_red.csv')


#%% For part 4
out = './Part4/'
ica = FastICA(random_state=5, n_components=4)
blocks_X2 = ica.fit_transform(blocks_X)
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
tmp.to_csv(out+'Pageblocks_ica_NN.csv')