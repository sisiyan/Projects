#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:32:13 2018

@author: syan
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_score

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
        train_subset = train_data.sample(frac = n, random_state = 1)
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



#Grid search with cross-validation to find the best condition for 'rbf' kernel
clf = SVC(kernel = 'rbf', cache_size = 500)
param_grid = {"C": [10, 1.5, 1.0, 0.5, 0.2, 0.1, 0.02],
              "gamma": [1, 0.1, 0.01, 0.001]}

grid_search = GridSearchCV(clf, param_grid=param_grid, cv = 5)

grid_search.fit(train_features, train_data[target])
report(grid_search.cv_results_)


# explore the Penalty parameter C of the error term, when using 'rbf' kernel function
C_range = [10, 1.5, 1.25, 1.0, 0.5, 0.2, 0.1, 0.02]
train_accuracy = []
cv_accuracy = []
train_precision = []
cv_precision = []
scoring = ['accuracy', 'precision']

for n in C_range:
    clf = SVC(C=n, kernel='rbf', gamma = 0.1, cache_size = 500)   
    train_acc,cv_acc, train_pc, cv_pc = cv_acc_precision(clf, train_features,\
                     train_data[target], scoring=scoring, cv = 5)

    train_accuracy.append(train_acc)
    cv_accuracy.append(cv_acc)
    train_precision.append(train_pc)
    cv_precision.append(cv_pc)
    
df_svc = pd.DataFrame({'train_accuracy': train_accuracy, \
                              'cv_accuracy': cv_accuracy,\
                              'train_precision': train_precision,\
                              'cv_precision': cv_precision, \
                              'C': C_range})    
print df_svc
ax = df_svc.plot(x='C', y=['train_accuracy','cv_accuracy','train_precision', 'cv_precision'],\
                 logx = True, title = "rbf SVC Complexity Curve: Penalty Parameter C", 
                 style = '.-', ylim = (0.5, 0.725))
ax.set_xlabel("C")
ax.set_ylabel("Accuracy/Precision")    
plt.savefig('Data1_SVM_rbf_complexityCurve.png')


#Grid search with cross-validation to find the best condition for the linearSVC
clf = LinearSVC(penalty='l2')
          
# specify parameters and distributions to sample from
param_grid = {"loss": ['squared_hinge', 'hinge'],
              "C": [1, 0.1, 0.05, 0.001],
              "max_iter": [100, 500, 1000, 2000, 5000]}

# run randomized search
grid_search = GridSearchCV(clf, param_grid=param_grid, cv = 5)

grid_search.fit(train_features, train_data[target])
print "l2 results"
report(grid_search.cv_results_, n_top=10)


#Grid search with cross-validation to find the best condition
clf = LinearSVC(penalty='l1', loss='squared_hinge', dual=False)
          
# specify parameters and distributions to sample from
param_grid = {"C": [1, 0.1, 0.05, 0.001],
              "max_iter": [100, 500, 1000, 2000, 5000]}

# run randomized search
grid_search = GridSearchCV(clf, param_grid=param_grid, cv = 5)

grid_search.fit(train_features, train_data[target])
print "l1 results"
report(grid_search.cv_results_)


# explore the Penalty parameter C of the error term, when using 'linear' kernel function
C_range = [1.5, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.001, 0.0001]
train_accuracy = []
cv_accuracy = []
train_precision = []
cv_precision = []
scoring = ['accuracy', 'precision']

for n in C_range:
    clf = LinearSVC(C = n, penalty='l2', loss='squared_hinge', dual=True)
    train_acc,cv_acc, train_pc, cv_pc = cv_acc_precision(clf, train_features,\
                     train_data[target], scoring=scoring, cv = 5)

    train_accuracy.append(train_acc)
    cv_accuracy.append(cv_acc)
    train_precision.append(train_pc)
    cv_precision.append(cv_pc)
    
df_svc = pd.DataFrame({'train_accuracy': train_accuracy, \
                              'cv_accuracy': cv_accuracy,\
                              'train_precision': train_precision,\
                              'cv_precision': cv_precision, \
                              'C': C_range})
print df_svc
ax = df_svc.plot(x='C', y=['train_accuracy','cv_accuracy','train_precision', 'cv_precision'],\
                 logx = True, title = "Linear SVC Complexity Curve: Penalty Parameter C",
                 style = '.-', ylim = (0.5, 0.725))
ax.set_xlabel("C")
ax.set_ylabel("Accuracy/Precision")    
plt.savefig('Data1_SVM_linear_complexityCurve.png')



# Plot the learning curve with different training sample size
# When using the 'rbf' kernel function, SVC classifier
df_loan_acc, df_loan_precison = training_size_acc(train_data, test_features, test_data[target], \
                        estimator = SVC(C= 1.0, kernel='rbf', gamma=0.1, cache_size = 500),\
                        scoring = ['accuracy', 'precision'])
print df_loan_acc
ax = df_loan_acc.plot(y=['train_accuracy','cv_accuracy','test_accuracy'],\
                      title = "RBF-SVM Learning Curve: Accuracy",\
                      ylim = (0.5, 0.725), style='.-')
ax.set_xlabel("Training size %")
ax.set_ylabel("Accuracy")  
plt.savefig('Data1_SVM_rbf_LearningCurve_acc.png')


# Plot the learning curve with different training sample size
# When using the LinearSVC classier
df_loan_acc, df_loan_precison = training_size_acc(train_data, test_features, test_data[target], \
                        estimator = LinearSVC(C = 0.5),\
                        scoring = ['accuracy', 'precision'])
print df_loan_acc
ax = df_loan_acc.plot(y=['train_accuracy','cv_accuracy','test_accuracy'],\
                      title = "Linear-SVM Learning Curve: Accuracy",\
                      ylim = (0.5, 0.725), style='.-')
ax.set_xlabel("Training size %")
ax.set_ylabel("Accuracy")  
plt.savefig('Data1_LinearSVM_LearningCurve_acc.png')

