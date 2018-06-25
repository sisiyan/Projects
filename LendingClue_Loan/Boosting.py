#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:42:43 2018

@author: syan
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# Helper function for scaling data
def scale_data(train_X, test_X):
    scaler = MinMaxScaler() 
    scaler.fit_transform(train_X)
    train_X = scaler.transform(train_X)  
    test_X = scaler.transform(test_X) 
    return train_X, test_X

#Helper function for cross validation 
def cv_acc_precision(estimator, train_X, train_Y, scoring=None, cv = 5):
    scores = cross_validate(estimator, train_X, train_Y, scoring=scoring,
                         cv=cv, return_train_score=True)
    train_acc = scores['train_accuracy'].mean()
    cv_acc = scores['test_accuracy'].mean()
    train_precision = scores['train_precision'].mean()
    cv_precision = scores['test_precision'].mean()
    return train_acc,cv_acc, train_precision, cv_precision

# Utility function to report best scores
def report(results, n_top=10):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
       
def training_size_acc(train_data, test_X, test_Y, estimator, scoring):
    train_size = np.array(range(10,110,10))
    train_accuracy = []
    cv_accuracy = []
    train_precision = []
    cv_precision = []
    test_accuracy = []
    test_precision = []
    
    for n in train_size *0.01:
        train_subset = train_data.sample(frac = n, random_state = 5 )
        train_subset_X = train_subset.iloc[:,0:26]
        train_subset_X = scale_data(train_subset_X, test_X)[0]
        
        train_acc,cv_acc, train_pc, cv_pc = cv_acc_precision(estimator, train_subset_X,\
                     train_subset[target], scoring=scoring, cv = 5)
        estimator.fit(train_subset_X, train_subset[target])
        
        test_acc = accuracy_score(test_Y, estimator.predict(test_X))
        test_pc = precision_score(test_Y, estimator.predict(test_X))
        train_accuracy.append(train_acc)
        cv_accuracy.append(cv_acc)
        train_precision.append(train_pc)
        cv_precision.append(cv_pc)
        test_accuracy.append(test_acc)
        test_precision.append(test_pc)
    
    df_loan_acc = pd.DataFrame({'train_accuracy': train_accuracy, \
                        'cv_accuracy': cv_accuracy, \
                         'test_accuracy': test_accuracy},\
                         index = train_size)
    df_loan_precison = pd.DataFrame({'train_precision': train_precision, \
                        'cv_precision': cv_precision, \
                         'test_precision': test_precision},\
                         index = train_size)
    return df_loan_acc, df_loan_precison   

         
# Load the lending club data
loans = pd.read_csv('lending_club_data.csv')

# Preprocessing the data
# reassign the labels to have +1 for a safe loan, and -1 for a risky (bad) loan.
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans.drop('bad_loans', axis=1, inplace = True)

# Use 4 categorical features for safe loan classification
features = ['loan_amnt',
            'annual_inc',
            'grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]

target = 'safe_loans'
loans = loans[features + [target]]

# Balance the number of safe loans and bad loans to avoid the safe loans dominate the training process
safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

# Since there are less risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
safe_loans = safe_loans_raw.sample(frac = percentage, random_state = 1)
risky_loans = risky_loans_raw
loans_data = risky_loans.append(safe_loans)

print "Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data))
print "Percentage of risky loans                :", len(risky_loans) / float(len(loans_data))
print "Total number of loans in our new dataset :", len(loans_data)

# Transform the categorical data into binary features
loans_features = loans_data[features]
loans_features = pd.get_dummies(loans_features)
loans_target = loans_data[target]
loans_data_binary = loans_features.join(loans_target)

train_data = loans_data_binary.sample(frac = 0.8, random_state = 1)
test_data = loans_data_binary.drop(train_data.index)
train_features = train_data.iloc[:, 0:26]
test_features = test_data.iloc[:, 0:26]
train_features, test_features = scale_data(train_features, test_features)


#Grid search with cross-validation to find the best condition
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))
# specify parameters and distributions to sample from
param_grid = {"n_estimators": [10, 20, 50, 100, 200],
              "learning_rate": [10.0, 1.0, 0.1, 0.01],
              "algorithm": ['SAMME.R', 'SAMME']}

grid_search = GridSearchCV(clf, param_grid=param_grid, cv = 5)

grid_search.fit(train_features, train_data[target])
report(grid_search.cv_results_, n_top = 10)

#Grid search with cross-validation to find the best condition
clf = AdaBoostClassifier(learning_rate = 1.0, algorithm = 'SAMME.R')
# specify parameters and distributions to sample from
param_grid = {"n_estimators": [10, 20, 50, 100, 200],
              "base_estimator":[DecisionTreeClassifier(max_depth=1), 
                                DecisionTreeClassifier(max_depth=2), 
                                DecisionTreeClassifier(max_depth=3)]}

grid_search = GridSearchCV(clf, param_grid=param_grid, cv = 5)

grid_search.fit(train_features, train_data[target])
report(grid_search.cv_results_, n_top = 10)


# Search for best value for n_estimators by cross-validation
n_estimators = [1, 5,10,20,50,100,200,300, 500]
train_accuracy = []
cv_accuracy = []
train_precision = []
cv_precision = []
scoring = ['accuracy', 'precision']

for n in n_estimators:
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
    n_estimators=n, algorithm='SAMME.R', learning_rate=1.0)
    train_acc,cv_acc, train_pc, cv_pc = cv_acc_precision(clf, train_features,\
                     train_data[target], scoring=scoring, cv = 5)
    train_accuracy.append(train_acc)
    cv_accuracy.append(cv_acc)
    train_precision.append(train_pc)
    cv_precision.append(cv_pc)
    
df_nEstimator = pd.DataFrame({'train_accuracy': train_accuracy, \
                              'cv_accuracy': cv_accuracy,\
                              'train_precision': train_precision,\
                              'cv_precision': cv_precision},\
                              index = n_estimators)
print df_nEstimator
ax = df_nEstimator.plot(y=['train_accuracy','cv_accuracy','train_precision', 'cv_precision'],\
                        title = "Complexity Curves: n_estimators", logx = True, style='.-', ylim= (0.5, 0.725))
ax.set_xlabel("n_estimators")
ax.set_ylabel("Accuracy/Precision")    
plt.savefig('Data1_Boost_nEstimator_complexityCurve.png')


# Plot the learning curve with different training sample size
# When using the max_depth = 1
df_loan_acc, df_loan_precison = training_size_acc(train_data, test_features, test_data[target], \
                        estimator = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
    n_estimators= 10, algorithm='SAMME.R', learning_rate=1.0),\
                        scoring = ['accuracy', 'precision'])
print df_loan_acc
ax = df_loan_acc.plot(y=['train_accuracy','cv_accuracy','test_accuracy'],\
                      title = "AdaBoosting Learning Curve: Accuracy",\
                      ylim = (0.5, 0.725), style = ".-")
ax.set_xlabel("Training size %")
ax.set_ylabel("Accuracy")  
plt.savefig('Data1_boost_LearningCurve_acc.png')
