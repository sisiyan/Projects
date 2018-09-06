#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 08:45:45 2018
Student: Si Yan
userID: syan62

"""
import pandas as pd
import numpy as np
import datetime as dt
import StrategyLearner as sl
import ManualStrategy as ms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from marketsimcode import compute_portvals


sv = 100000
qlearner = sl.StrategyLearner(verbose = False, impact = 0.000) # constructor
qlearner.addEvidence(symbol="JPM",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=sv)
q_trades = qlearner.testPolicy(symbol="JPM",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=sv)
q_portvals, q_cr, q_stdev, q_mean = compute_portvals(q_trades, start_val = sv, commission=0, impact=0) 

portvals = np.zeros((len(q_portvals), 7))
cr = []
stdev = []
mean = []

for i in range(0,7):
    qlearner = sl.StrategyLearner(verbose = False, impact = 0.000) # constructor
    qlearner.addEvidence(symbol="JPM",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=sv)
    q_trades = qlearner.testPolicy(symbol="JPM",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=sv)
    q_portvals, q_cr, q_stdev, q_mean = compute_portvals(q_trades, start_val = sv, commission=0, impact=0) 
    portvals[:,i] = q_portvals
    cr.append(q_cr)
    stdev.append(q_stdev)
    mean.append(q_mean)
 
median_trial = np.argsort(portvals[-1])[len(portvals[-1])//2]   
med_portvals = portvals[:, median_trial]
med_cr = cr[median_trial]
med_stdev = stdev[median_trial]
med_mean = mean[median_trial]

print "Strategy Learner Results in In-sample period:"
print "Start value of portfolio on 2008-01-01: %f" %(sv)
print "Final value of portfolio on 2009-12-31: %f" %(med_portvals[-1])
print "The Cumulative return of portfolio: %f" %med_cr
print "The Stdev of daily returns of portfolio: %f" %med_stdev
print "The Mean of daily returns of portfolio: %f" %med_mean 


m_St = ms.ManualStrategy() # constructor
ms_trades = m_St.testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = sv)
ms_portvals, ms_cr, ms_stdev, ms_mean = compute_portvals(ms_trades, start_val = sv, commission=0, impact=0) 

print "Manual Strategy Results in In-sample period:"
print "Start value of portfolio on 2008-01-01: %f" %(sv)
print "Final value of portfolio on 2009-12-31: %f" %(ms_portvals[-1])
print "The Cumulative return of portfolio: %f" %ms_cr
print "The Stdev of daily returns of portfolio: %f" %ms_stdev
print "The Mean of daily returns of portfolio: %f" %ms_mean 

#benchmark
bm_trades = pd.DataFrame(data = 0, index = q_trades.index, columns = ["JPM"])
bm_trades["JPM"][0] = 1000
#print bm_trades.head()
bm_portvals, bm_cr, bm_stdev, bm_mean = compute_portvals(df_trades = bm_trades, start_val = sv, commission=0.0, impact=0.0)
print "The benchmark Results in In-sample period::"
print "Start value of benchmark on 2008-01-01: %f" %(sv)
print "Final value of benchmark on 2009-12-31: %f" %(bm_portvals[-1])
print "The Cumulative return of benchmark: %f" %bm_cr
print "The Stdev of daily returns of benchmark: %f" %bm_stdev
print "The Mean of daily returns of benchmark: %f" %bm_mean

#%%

plt.plot(q_portvals.index, med_portvals/med_portvals[0], color = "red")
plt.plot(ms_portvals.index, ms_portvals/ms_portvals[0], color = "black")
plt.plot(bm_portvals.index, bm_portvals/bm_portvals[0], color = "blue")
#plt.plot(p_sma.index, p_sma, label = "Price/SMA ratio")
plt.legend(["Strategy learner", "Manual strategy","Benchmark"],loc = 'upper left')
plt.title("Strategy Learner and Manual Strategy Comparison: In Sample")
plt.ylabel("Normalized Portfolio Value")
plt.xlabel("Time")
plt.xlim(dt.datetime(2008,1,1), dt.datetime(2009,12,31))
#plt.ylim(0.5, 3.5)
plt.tick_params(axis='both', which='major', labelsize=9)
plt.savefig("Exp1_sl_ms_bm.png", dpi=300)
plt.close()

#%% Out of sample of strategy learner
#sv = 100000
#qlearner = sl.StrategyLearner(verbose = False, impact = 0.000) # constructor
#qlearner.addEvidence(symbol="JPM",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=sv)
#q_trades = qlearner.testPolicy(symbol="JPM",sd=dt.datetime(2010,1,1),ed=dt.datetime(2011,12,31),sv=sv)
#q_portvals, q_cr, q_stdev, q_mean = compute_portvals(q_trades, start_val = sv, commission=0, impact=0) 
#
#print "Strategy Learner Results in In-sample period:"
#print "Start value of portfolio on 2008-01-01: %f" %(sv)
#print "Final value of portfolio on 2009-12-31: %f" %(q_portvals[-1])
#print "The Cumulative return of portfolio: %f" %q_cr
#print "The Stdev of daily returns of portfolio: %f" %q_stdev
#print "The Mean of daily returns of portfolio: %f" %q_mean 
