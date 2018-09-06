#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 08:36:21 2018
Student: Si Yan
userID: syan62
"""

"""MC2-P1: Market simulator.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data


def compute_portvals(df_trades, start_val = 100000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    
    nrow = df_trades.shape[0]
    start_date = df_trades.index[0]
    end_date = df_trades.index[nrow - 1]
    symbols = list(df_trades.columns)
    symbol = str(df_trades.columns[0])
    #print type(symbol)
    #print symbols
    #trades_columns.append('cash')
    # Create the dataframe of stock prices
    prices_df = get_data(symbols, pd.date_range(start_date, end_date), addSPY=True, colname = 'Adj Close')
    prices_df['cash'] = 1.0
    prices_df = prices_df.drop(['SPY'], axis = 1)
    prices_df = prices_df.fillna(method = 'ffill')
    prices_df = prices_df.fillna(method = 'bfill')
    
    
    trades_df = pd.DataFrame(data = df_trades[symbol], index = prices_df.index, columns = symbols)
    trades_df['cash'] = 0.0
  
    buys = trades_df[symbol].loc[trades_df[symbol] >0]
    buys_price = prices_df[symbol].loc[trades_df[symbol] >0]
    trades_df['cash'].loc[trades_df[symbol] >0] = -(buys * buys_price * (1+impact) + commission)
    
    sells = trades_df[symbol].loc[trades_df[symbol]< 0]
    sells_price = prices_df[symbol].loc[trades_df[symbol]< 0]
    trades_df['cash'].loc[trades_df[symbol] < 0] = -sells * sells_price * (1-impact) - commission
    
    holding_df = pd.DataFrame(data = 0.0, index = prices_df.index, columns = symbols)
    holding_df['cash'] = 0
    
    holding_df['cash'].iloc[0] = start_val
    holding_df.iloc[0] = holding_df.iloc[0] + trades_df.iloc[0]
    for i in range(1, holding_df.shape[0]):
        holding_df.iloc[i] = holding_df.iloc[i -1] + trades_df.iloc[i]
        
    value_df = prices_df * holding_df
    
    portvals = value_df.sum(axis = 1)
    #compute cumulative return
    cr = portvals[-1]/start_val - 1
    #compute stdev and mean of daily returns
    dr = portvals/portvals.shift(1) - 1
    stdev = dr[1:].std()
    mean = dr[1:].mean()
    #print portvals.head(20)
    #print dr.head(20)
    #print cr, stdev, mean
    #print prices_df.loc['2008-01-01':'2008-03-01']
    #print trades_df.loc['2008-09-15':'2008-10-15']
    #print holding_df.loc['2008-01-01':'2008-03-01']
    #print value_df.loc['2008-01-01':'2008-03-01']
    #print portvals.loc['2008-08-05':'2008-08-15']
    
    return portvals, cr, stdev, mean

def author():
    return 'syan62'


#def test_code():
#    # this is a helper function you can use to test your code
#    # note that during autograding his function will not be called.
#    # Define input parameters
#    #bps = BestPossibleStrategy()  
#    import BestPossibleStrategy as bps
#    bpsTrades = bps.BestPossibleStrategy()
#
#    df_trades = bpsTrades.testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) 
#
#    # Process orders
#    bps_portvals, bps_cr, bps_stdev, bps_mean = compute_portvals(df_trades = df_trades, start_val = 100000, commission=0.00, impact=0.00)
#    if isinstance(bps_portvals, pd.DataFrame):
#        print "Returns a DataFrame correctly." # just get the first column
#    else:
#        print "warning, code did not return a DataFrame"
#    print "The best possible strategy:"
#    print "Start value of portfolio on 2008-01-01: %f" %(bps_portvals[0])
#    print "Final value of portfolio on 2009-12-31: %f" %(bps_portvals[-1])
#    print "The Cumulative return of portfolio: %f" %bps_cr
#    print "The Stdev of daily returns of portfolio: %f" %bps_stdev
#    print "The Mean of daily returns of portfolio: %f" %bps_mean
    

if __name__ == "__main__":
    print "This is a marketsimcode"
    #test_code()
