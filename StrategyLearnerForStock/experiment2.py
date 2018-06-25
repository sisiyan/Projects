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
import random



sv = 100000
#impact = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.6, 1.0]
impact = [0.0, 0.002, 0.004, 0.006, 0.008, 0.01]
final_portvals = []
portvals = []
crs = []
stdevs = []
means = []
numOfTrades = []
    
for i in range(len(impact)):
    qlearner = sl.StrategyLearner(verbose = False, impact = impact[i]) # constructor
    qlearner.addEvidence(symbol="JPM",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=sv)
    q_trades = qlearner.testPolicy(symbol="JPM",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=sv)
    q_portvals, q_cr, q_stdev, q_mean = compute_portvals(q_trades, start_val = sv, commission=0, impact=impact[i]) 
    portvals.append(q_portvals)
    final_portvals.append(q_portvals[-1])
    crs.append(q_cr)
    stdevs.append(q_stdev)
    means.append(q_mean)
    numOfTrades.append(len(q_trades[q_trades["JPM"] != 0]))
    
    
    plt.plot(q_portvals.index, q_portvals/q_portvals[0], label = "impact= %.3f" %impact[i])
    
    #plt.plot(p_sma.index, p_sma, label = "Price/SMA ratio")
    plt.legend(loc = 'upper left', fontsize = 7)
    plt.title("Strategy Learner Performance with Different Impact")
    plt.xlabel("Time", fontsize = 8)
    plt.ylabel("Normalized Portfolio Value", fontsize = 8)
    
    plt.xlim(dt.datetime(2008,1,1), dt.datetime(2009,12,31))
    #plt.ylim(0.5, 3.5)
    plt.tick_params(axis='both', which='major', labelsize=7)
    
          
plt.savefig("Exp2_sl_impacts.png", dpi=300)

#%%
impact = [0.0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.05, 0.1, 0.3, 0.6]
final_portvals = []
portvals = []
crs = []
stdevs = []
means = []
numOfTrades = []

#ms_final_portvals = []
#ms_crs = []
#ms_numOfTrades = []
for i in range(len(impact)):
    
#    m_St = ms.ManualStrategy() # constructor
#    ms_trades = m_St.testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = sv)
#    ms_portvals, ms_cr, ms_stdev, ms_mean = compute_portvals(ms_trades, start_val = sv, commission=0, impact=impact[i])
#    ms_numOfTrades.append(len(ms_trades[ms_trades["JPM"] != 0]))
#    ms_final_portvals.append(ms_portvals[-1])
#    ms_crs.append(ms_cr)
    
    qlearner = sl.StrategyLearner(verbose = False, impact = impact[i]) # constructor
    qlearner.addEvidence(symbol="JPM",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=sv)
    q_trades = qlearner.testPolicy(symbol="JPM",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=sv)
    q_portvals, q_cr, q_stdev, q_mean = compute_portvals(q_trades, start_val = sv, commission=0, impact=impact[i]) 
    portvals.append(q_portvals)
    final_portvals.append(q_portvals[-1])
    crs.append(q_cr)
    stdevs.append(q_stdev)
    means.append(q_mean)
    numOfTrades.append(len(q_trades[q_trades["JPM"] != 0]))
    
    
df_results = pd.DataFrame({'sl_Trade_Times':numOfTrades, 'sl_final_portvals': final_portvals, \
                            'sl_CR': crs, \
                            'Stddev': stdevs,\
                            'means': means},\
                             index = impact)    

df_results.to_csv("exp2_results.csv")
ax = df_results.plot(y=['sl_Trade_Times'], style = ".-", title = "Trade Times at Different Impact")
ax.set_xlabel("Impact")
ax.set_ylabel("Number of Trades")  
plt.savefig('Exp2_trade_times.png', dpi=300)

ax2 = df_results.plot(y=['sl_CR'], style = ".-", title = "Cumulative Return at Different Impact")
ax2.set_xlabel("Impact")
ax2.set_ylabel("Cumulative Return")  
plt.savefig('Exp2_CR.png', dpi=300)

#%%



#qlearner = sl.StrategyLearner(verbose = False, impact = impact) # constructor
#qlearner.addEvidence(symbol="JPM",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=sv)
#q_trades = qlearner.testPolicy(symbol="JPM",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=sv)
#q_portvals, q_cr, q_stdev, q_mean = compute_portvals(q_trades, start_val = sv, commission=0, impact=impact) 
#
#print "Strategy Learner Results in In-sample period:"
#print "Start value of portfolio on 2008-01-01: %f" %(sv)
#print "Final value of portfolio on 2009-12-31: %f" %(q_portvals[-1])
#print "The Cumulative return of portfolio: %f" %q_cr
#print "The Stdev of daily returns of portfolio: %f" %q_stdev
#print "The Mean of daily returns of portfolio: %f" %q_mean 
#
#
#m_St = ms.ManualStrategy() # constructor
#ms_trades = m_St.testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = sv)
#ms_portvals, ms_cr, ms_stdev, ms_mean = compute_portvals(ms_trades, start_val = sv, commission=0, impact=impact) 
#
#print "Manual Strategy Results in In-sample period:"
#print "Start value of portfolio on 2008-01-01: %f" %(sv)
#print "Final value of portfolio on 2009-12-31: %f" %(ms_portvals[-1])
#print "The Cumulative return of portfolio: %f" %ms_cr
#print "The Stdev of daily returns of portfolio: %f" %ms_stdev
#print "The Mean of daily returns of portfolio: %f" %ms_mean 
#
##benchmark
#bm_trades = pd.DataFrame(data = 0, index = q_trades.index, columns = ["JPM"])
#bm_trades["JPM"][0] = 1000
##print bm_trades.head()
#bm_portvals, bm_cr, bm_stdev, bm_mean = compute_portvals(df_trades = bm_trades, start_val = sv, commission=0.0, impact=impact)
#print "The benchmark Results in In-sample period::"
#print "Start value of benchmark on 2008-01-01: %f" %(sv)
#print "Final value of benchmark on 2009-12-31: %f" %(bm_portvals[-1])
#print "The Cumulative return of benchmark: %f" %bm_cr
#print "The Stdev of daily returns of benchmark: %f" %bm_stdev
#print "The Mean of daily returns of benchmark: %f" %bm_mean
#
##%%
#
#plt.plot(q_portvals.index, q_portvals/q_portvals[0], color = "red")
#plt.plot(ms_portvals.index, ms_portvals/ms_portvals[0], color = "black")
#plt.plot(bm_portvals.index, bm_portvals/bm_portvals[0], color = "blue")
##plt.plot(p_sma.index, p_sma, label = "Price/SMA ratio")
#plt.legend(["Strategy learner", "Manual strategy","Benchmark"],loc = 'upper left')
#plt.title("Strategy Learner and Manual Strategy Comparison: In Sample")
#plt.ylabel("Normalized Portfolio Value")
#plt.xlabel("Time")
#plt.xlim(dt.datetime(2008,1,1), dt.datetime(2009,12,31))
##plt.ylim(0.5, 3.5)
#plt.tick_params(axis='both', which='major', labelsize=9)
#plt.savefig("Exp2_sl_ms_bm.png", dpi=300)
#plt.close()