#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 15:17:02 2018

@author: syan
"""

#%% Imports
import pandas as pd
import numpy as np
from time import time
from helpers import nn_reg,nn_arch
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import ImportanceSelect
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

out = './Part2/RF/'

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
#clusters =  [2,5,10,15,20,25,30,35,40]
#dims = [2,5,10,15,20,25,30,35,40,45,50,55,60]
    
#%% Run RF to select features
    
rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5)
fi_RF_loans = rfc.fit(loans_X,loans_Y).feature_importances_ 

plt.plot(range(26),fi_RF_loans, marker = 'o',markersize=4, linestyle = "-")
plt.title("Loans Feature Importance by RF")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.savefig(out+'Loans_part2_RF.png')
plt.close()

tmp1 = pd.Series(np.sort(fi_RF_loans)[::-1])
tmp1.to_csv(out+'fi_RF_loans.csv')

fi_RF_pageblocks = rfc.fit(blocks_X,blocks_Y).feature_importances_ 

plt.plot(range(10),fi_RF_pageblocks, marker = 'o',markersize=4, linestyle = "-")
plt.title("Pageblocks Feature Importance by RF")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.savefig(out+'Pageblock_part2_RF.png')
plt.close()

tmp2 = pd.Series(np.sort(fi_RF_pageblocks)[::-1])
tmp2.to_csv(out+'fi_RF_pageblocks.csv')

#%% Validation for part2

filtr = ImportanceSelect(rfc)

dims1 = [2, 4, 5, 10, 15, 20, 26]
grid ={'filter__n':dims1}
mlp = MLPClassifier(solver='lbfgs', activation='identity', alpha=0.1,
                     hidden_layer_sizes=(50,),
                    max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('filter',filtr),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(loans_X,loans_Y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Loans_RF_dim_red.csv')

dims2 = [2, 3, 4, 5, 6,7,8,9,10]
grid ={'filter__n':dims2}  
mlp = MLPClassifier(solver='lbfgs', activation='logistic', alpha=0.1,
                     hidden_layer_sizes=(50,),
                    max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('filter',filtr),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(blocks_X,blocks_Y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'pageblocks_RF_dim_red.csv')


#%% For part 4
out = './Part4/'

rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5)
dim = 5
filtr = ImportanceSelect(rfc,dim)
blocks_X2 = filtr.fit_transform(blocks_X,blocks_Y)


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
tmp.to_csv(out+'Pageblocks_RF_NN.csv')