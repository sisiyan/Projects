#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:10:53 2018

@author: syan
"""

 
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
#import graphviz
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_score

## Helper function for scaling data
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
        train_subset = train_data.sample(frac = n, random_state = 4)
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

#Define the score types are used in the dataset classification
scoring = ['accuracy', 'precision']

#Grid search with cross-validation to find the best condition
clf = MLPClassifier(max_iter = 500)
param_dist = { "hidden_layer_sizes": [(50,),(100,),(10,10), (50,50)],    
            "activation": ['identity', 'logistic'],
              "solver": ['lbfgs', 'sgd', 'adam'],
              "alpha": np.power(10.0, range(-3, 3, 1)),
              "batch_size": range(100, 1000, 200)}
# run randomized search
n_iter_search = 50
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
              n_iter=n_iter_search, cv= 5)

random_search.fit(train_features, train_data[target])
report(random_search.cv_results_, n_top = 20)


# Model complexity curve 
max_iter = [100, 150, 200, 250, 500, 1000]
train_accuracy = []
cv_accuracy = []
train_precision = []
cv_precision = []

for n in max_iter:
    clf = MLPClassifier(solver='lbfgs', activation='identity', alpha=0.01,
                     hidden_layer_sizes=(n, ), random_state=2, max_iter = n)
    train_acc,cv_acc, train_pc, cv_pc = cv_acc_precision(clf, train_features,\
                     train_data[target], scoring=scoring, cv = 5)
    train_accuracy.append(train_acc)
    cv_accuracy.append(cv_acc)
    train_precision.append(train_pc)
    cv_precision.append(cv_pc)
    
df_maxIter = pd.DataFrame({'train_accuracy': train_accuracy, \
                              'cv_accuracy': cv_accuracy,\
                              'train_precision': train_precision,\
                              'cv_precision': cv_precision},\
                              index = max_iter)
print df_maxIter
ax = df_maxIter.plot(y=['train_accuracy','cv_accuracy','train_precision', 'cv_precision'],\
                        title = "Complexity Curve: Max Iterations", 
                        style = ".-", ylim = (0.5, 0.725))
ax.set_xlabel("Max Iterations")
ax.set_ylabel("Accuracy/Precision")  
plt.savefig('Data1_NN_maxIter_complexityCurve.png')


# Model complexity curve
hidden_layer_size = [1, 2, 5, 10, 25, 50, 100, 200, 500]
train_accuracy = []
cv_accuracy = []
train_precision = []
cv_precision = []

for n in hidden_layer_size:
    clf = MLPClassifier(solver='lbfgs', activation='identity', alpha=0.01,
                     hidden_layer_sizes=(n), random_state=2, max_iter = 250)
    train_acc,cv_acc, train_pc, cv_pc = cv_acc_precision(clf, train_features,\
                     train_data[target], scoring=scoring, cv = 5)

    train_accuracy.append(train_acc)
    cv_accuracy.append(cv_acc)
    train_precision.append(train_pc)
    cv_precision.append(cv_pc)
    
df_layerSize = pd.DataFrame({'train_accuracy': train_accuracy, \
                              'cv_accuracy': cv_accuracy,\
                              'train_precision': train_precision,\
                              'cv_precision': cv_precision},\
                              index = hidden_layer_size)
print df_layerSize
ax = df_layerSize.plot(y=['train_accuracy','cv_accuracy','train_precision', 'cv_precision'],\
                        title = "Complexity Curve: hidden_layer_size", 
                        style = ".-", ylim = (0.5, 0.725), logx = True)
ax.set_xlabel("Hidden_layer_size")
ax.set_ylabel("Accuracy/Precision")  
plt.savefig('Data1_NN_layerSize_complexityCurve.png')

print "Check neural networks with more than one hidden layers:"
clf = MLPClassifier(solver='lbfgs', activation='identity', alpha=0.01,
                     hidden_layer_sizes=(50, 50), random_state=1, max_iter = 250)
print "Two hidden layers, with 50 and 50 neurons on each layer:"
print cv_acc_precision(clf, train_features,\
                     train_data[target], scoring=scoring, cv = 5)
clf = MLPClassifier(solver='lbfgs', activation='identity', alpha=0.01,
                     hidden_layer_sizes=(50, 10), random_state=1, max_iter = 250)
print "Two hidden layers, with 50 and 10 neurons on each layer:"
print cv_acc_precision(clf, train_features,\
                     train_data[target], scoring=scoring, cv = 5)

clf = MLPClassifier(solver='lbfgs', activation='identity', alpha=0.01,
                     hidden_layer_sizes=(50, 50, 50), random_state=1, max_iter = 250)
print "Three hidden layers, with 50, 50 and 50 neurons on each layer:"
print cv_acc_precision(clf, train_features,\
                     train_data[target], scoring=scoring, cv = 5)


# Plot the learning curve with different training sample size
# When using the max_depth = 1
df_loan_acc, df_loan_precison = training_size_acc(train_data, test_features, test_data[target], \
                        estimator = MLPClassifier(solver='lbfgs', activation='identity', alpha=0.01,
                     hidden_layer_sizes=(5), random_state=2, max_iter = 250),\
                        scoring = ['accuracy', 'precision'])
print df_loan_acc
ax = df_loan_acc.plot(y=['train_accuracy','cv_accuracy','test_accuracy'],\
                      title = "Neural Network Learning Curve: Accuracy",\
                      ylim = (0.5, 0.725), style = '.-')
ax.set_xlabel("Training size %")
ax.set_ylabel("Accuracy")  
plt.savefig('Data1_NN_LearningCurve_acc.png')
