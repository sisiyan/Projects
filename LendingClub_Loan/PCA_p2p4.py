#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 09:55:41 2018

@author: syan
"""

#%% Imports
import pandas as pd
import numpy as np
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import  nn_arch,nn_reg
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

out = './Part2/PCA/'

np.random.seed(0)

loans = pd.read_csv('./lending_club_scaled.csv')
loans_X = loans.drop('safe_loans',1).copy().values
loans_Y = loans['safe_loans'].copy().values

pca = PCA(random_state=5)
pca.fit(loans_X)
loans_eigenvalues = pd.Series(data = pca.explained_variance_,index = range(1,27))
loans_eigenvalues.to_csv(out+'./pca_loans_eigenvalues.csv')



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

pca = PCA(random_state=5)
pca.fit(blocks_X)
pageblocks_eigenvalues = pd.Series(data = pca.explained_variance_,index = range(1,11))
pageblocks_eigenvalues.to_csv(out+'pca_pb_eigenvalues.csv')


#%% Plot pca eigenvalues for part 2
ax = loans_eigenvalues.plot(
                      title = "Loans PCA eigenvalues",\
                      style = '.')
ax.set_xlabel("Dimension")
ax.set_ylabel("Eigenvalue")  
plt.savefig(out+'Loans_part2_pcaEV.png', dpi=300)
plt.close()

ax = pageblocks_eigenvalues.plot(
                      title = "Pageblocks PCA eigenvalues",\
                      style = '.')
ax.set_xlabel("Dimension")
ax.set_ylabel("Eigenvalue")  
plt.savefig(out+'pageblocks_part2_pcaEV.png', dpi=300)
plt.close()

#%% Validation for Part2

dims1 = [2,4,5,7,10,15,20,22, 26]

grid ={'pca__n_components':dims1,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
pca = PCA(random_state=5)       
mlp = MLPClassifier(solver='lbfgs', activation='identity', 
                    max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(loans_X,loans_Y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Loan_dim_red_ownNN.csv')


dims2 = [2,3,4,5,6,7,8,9,10]
#dims2 = [2,10]
grid ={'pca__n_components':dims2}
pca = PCA(random_state=5)       
mlp = MLPClassifier(solver='lbfgs', activation='logistic', alpha=0.1,
                     hidden_layer_sizes=(50,),
                    max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(blocks_X,blocks_Y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Pageblocks_dim_red_ownNN2.csv')

#%% For clock time part 4
out = './Part4/'
pca = PCA(random_state=5, n_components = 6)
blocks_X2 = pca.fit_transform(blocks_X)
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
tmp.to_csv(out+'Pageblocks_pca_NN.csv')