#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 14:08:30 2018
Student: Si Yan
userID: syan62
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
from marketsimcode import compute_portvals
from indicators import p_sma_ratio, stochastic_K_D, bband
 
import matplotlib.pyplot as plt

class ManualStrategy():
    def __init__(self):
        self.netHolding = 0;
        self.sv = 0;
    
    def testPolicy(self, symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000):
        self.sv = sv
        #Read in the stock price
        if type(symbol) is str:
            symbols = list()
            symbols.append(symbol)
        else:
            symbols = symbol
        prices_df = get_data(symbols, pd.date_range(sd, ed), addSPY=True, colname = 'Adj Close')
        #prices_df['cash'] = 1.0
        #prices_df = prices_df.drop(['SPY'], axis = 1)
        prices_df = prices_df.fillna(method = 'ffill')
        prices_df = prices_df.fillna(method = 'bfill')
        
        lookback_days = 30
        #date_range = prices_df.index[(lookback_days-1):]
        #Create an empty dataframe df_trades
        df_trades = pd.DataFrame(data = 0, index = prices_df.index, columns = symbols)
        #print df_trades.head(20)
        sma, p_sma = p_sma_ratio(prices_df, lookback = lookback_days)
        K, D = stochastic_K_D(prices_df, lookback = lookback_days)
        BB, upperband, lowerband = bband(prices_df, lookback = lookback_days)
        
        
        for i in range(1, df_trades.shape[0]-(lookback_days-1)):
            #The long entry
            if BB[symbol][i-1] <= -1.01 and p_sma[symbol][i] > p_sma[symbol][i-1] and D[symbol][i-1] < 20:
                if self.netHolding == -1000:
                    df_trades.iloc[i+lookback_days-1] = 2000
                elif self.netHolding == 0: 
                    df_trades.iloc[i+lookback_days-1] = 1000
            #The short entry        
            elif BB[symbol][i-1] >= 1.01 and p_sma[symbol][i] < p_sma[symbol][i-1] and D[symbol][i-1] > 80:  
                # 
                if self.netHolding == 1000:
                    df_trades.iloc[i+lookback_days-1] = -2000
                elif self.netHolding == 0:
                    df_trades.iloc[i+lookback_days-1] = -1000
                    
            #The exit position  
            elif (BB[symbol][i-1] >= 0.95 and BB[symbol][i-1] <1.0) or (BB[symbol][i-1] > -1.0 and BB[symbol][i-1] <= -0.95):
                df_trades.iloc[i+lookback_days-1] = -self.netHolding
 
            self.netHolding += df_trades.iloc[i + lookback_days-1].values    
            
        return df_trades
    
def main():
    ms = ManualStrategy()
    sv = 100000
    df_trades = ms.testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = sv)
    
    #print df_trades.head(20)
    #print df_trades.loc['2008-01-01':'2008-03-01']
    ms_portvals, ms_cr, ms_stdev, ms_mean = compute_portvals(df_trades, start_val = sv, commission=9.95, impact=0.005)
    print "In sample period:"
    print "Manual strategy"
    print "Start value of portfolio on 2008-01-01: %f" %(sv)
    print "Final value of portfolio on 2009-12-31: %f" %(ms_portvals[-1])
    print "The Cumulative return of portfolio: %f" %ms_cr
    print "The Stdev of daily returns of portfolio: %f" %ms_stdev
    print "The Mean of daily returns of portfolio: %f" %ms_mean
    #print ms_portvals.loc['2008-01-01':'2008-03-01']
    
    
    #benchmark
    bm_trades = pd.DataFrame(data = 0, index = df_trades.index, columns = ["JPM"])
    bm_trades["JPM"][0] = 1000
    #print bm_trades.head()
    bm_portvals, bm_cr, bm_stdev, bm_mean = compute_portvals(df_trades = bm_trades, start_val = sv, commission=9.95, impact=0.005)
    print "The benchmark:"
    print "Start value of benchmark on 2008-01-01: %f" %(sv)
    print "Final value of benchmark on 2009-12-31: %f" %(bm_portvals[-1])
    print "The Cumulative return of benchmark: %f" %bm_cr
    print "The Stdev of daily returns of benchmark: %f" %bm_stdev
    print "The Mean of daily returns of benchmark: %f" %bm_mean
    #print bm_portvals
    
    
    plt.plot(ms_portvals.index, ms_portvals/ms_portvals[0], color = "black")
    plt.plot(bm_portvals.index, bm_portvals/bm_portvals[0], color = "blue")
    #plt.plot(p_sma.index, p_sma, label = "Price/SMA ratio")
    plt.legend(["Manual strategy","Benchmark"],loc = 'upper left')
    
    for i in range(df_trades.shape[0]-1):
        if ms_portvals[i+1] - ms_portvals[i] != 0:
            if df_trades["JPM"].iloc[i] > 0:
                plt.axvline(df_trades.index[i], linewidth = 1, color = 'green', linestyle='dashed')
            elif df_trades["JPM"].iloc[i] < 0: 
                plt.axvline(df_trades.index[i], linewidth = 1, color = 'red', linestyle='dashed')
    plt.grid(color='gray', linestyle='--', axis='y', linewidth=1, alpha = 0.4)            
    plt.title("Manual Strategy vs Benchmark: In Sample")
    plt.ylabel("Normalized Portfolio Value")
    plt.xlabel("Time")
    plt.xlim(dt.datetime(2008,1,1), dt.datetime(2009,12,31))
    plt.ylim(0.7, 1.5)
    plt.tick_params(axis='both', which='major', labelsize=9)
    plt.savefig("ManualStrategy_train.png", dpi=300)
    plt.close()
    
    df_trades_test = ms.testPolicy(symbol = "JPM", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = sv)   
    ms_portvals, ms_cr, ms_stdev, ms_mean = compute_portvals(df_trades = df_trades_test, start_val = sv, commission=9.95, impact=0.005)
    print "Out of sample period:"
    print "Manual strategy"
    print "Start value of portfolio on 2010-01-01: %f" %(sv)
    print "Final value of portfolio on 2011-12-31: %f" %(ms_portvals[-1])
    print "The Cumulative return of portfolio: %f" %ms_cr
    print "The Stdev of daily returns of portfolio: %f" %ms_stdev
    print "The Mean of daily returns of portfolio: %f" %ms_mean
    
    #benchmark
    bm_trades = pd.DataFrame(data = 0, index = df_trades_test.index, columns = ["JPM"])
    bm_trades["JPM"][0] = 1000
    bm_portvals, bm_cr, bm_stdev, bm_mean = compute_portvals(df_trades = bm_trades, start_val = sv, commission=9.95, impact=0.005)
    print "The benchmark:"
    print "Start value of benchmark on 2010-01-01: %f" %(sv)
    print "Final value of benchmark on 2011-12-31: %f" %(bm_portvals[-1])
    print "The Cumulative return of benchmark: %f" %bm_cr
    print "The Stdev of daily returns of benchmark: %f" %bm_stdev
    print "The Mean of daily returns of benchmark: %f" %bm_mean
    
    
    plt.plot(ms_portvals.index, ms_portvals/ms_portvals[0], color = "black")
    plt.plot(bm_portvals.index, bm_portvals/bm_portvals[0], color = "blue")
    #plt.plot(p_sma.index, p_sma, label = "Price/SMA ratio")
    plt.legend(["Manual strategy","Benchmark"],loc = 'upper left')
    
    for i in range(df_trades_test.shape[0]-1):
        if ms_portvals[i+1] - ms_portvals[i] != 0:
            if df_trades_test["JPM"].iloc[i] > 0:
                plt.axvline(df_trades_test.index[i], linewidth = 1, color = 'green', linestyle='dashed')
            elif df_trades_test["JPM"].iloc[i] < 0: 
                plt.axvline(df_trades_test.index[i], linewidth = 1, color = 'red', linestyle='dashed')
    plt.grid(color='gray', linestyle='--', axis='y', linewidth=1, alpha = 0.4)            
    plt.title("Manual Strategy vs Benchmark: Out of Sample")
    plt.ylabel("Normalized Portfolio Value")
    plt.xlabel("Time")
    plt.xlim(dt.datetime(2010,1,1), dt.datetime(2011,12,31))
    plt.ylim(0.7, 1.5)
    plt.tick_params(axis='both', which='major', labelsize=9)
    plt.savefig("ManualStrategy_test.png", dpi=300)
    plt.close()
    
if __name__ == "__main__":
    main()       