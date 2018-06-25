#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:04:36 2018
Student: Si Yan
userID: syan62
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def p_sma_ratio(dataframe, lookback = 30):
    sma = dataframe.rolling(window = lookback).mean()
    sma = sma[(lookback-1):]
    
    p_sma = dataframe.iloc[(lookback-1):]/sma
    return sma, p_sma

def stochastic_K_D(dataframe, lookback = 30):    
    K_df = pd.DataFrame(data = 0.0, index = dataframe.index[(lookback -1): dataframe.shape[0]], 
           columns = dataframe.columns)
    
    for i in range(lookback -1, dataframe.shape[0]):
        close_price =dataframe.iloc[i]
        L = np.amin(dataframe.iloc[i+1-lookback:i+1], axis = 0)
        H = np.amax(dataframe.iloc[i+1-lookback:i+1], axis = 0)
        percentK = 100.0*(close_price- L) / (H-L)
        K_df.iloc[i-(lookback -1)] = percentK
   
    D_df = K_df.rolling(window = 3).mean()
    return  K_df, D_df    

def bband(dataframe, lookback = 30):
    nor_dataframe = dataframe/dataframe.iloc[0]
    rstd = nor_dataframe.rolling(window = lookback).std()
    sma = nor_dataframe.rolling(window = lookback).mean()
    rstd = rstd[(lookback-1):]
    sma = sma[(lookback-1):]
    BB = (nor_dataframe[(lookback-1):] - sma) / (2 * rstd)
    upper_band = (sma + 2 * rstd)[(lookback-1):]
    lower_band = (sma - 2 * rstd)[(lookback-1):]
    return BB, upper_band, lower_band
    

def main():
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,12,31)
    prices_df = get_data(["JPM"], pd.date_range(start_date, end_date), addSPY=True, colname = 'Adj Close')
    prices_df = prices_df.drop(['SPY'], axis = 1)
    prices_df = prices_df.fillna(method = 'ffill')
    prices_df = prices_df.fillna(method = 'bfill')
    
    volume_df = get_data(["JPM"], pd.date_range(start_date, end_date), addSPY=True, colname = 'Volume')
    volume_df = volume_df.drop(['SPY'], axis = 1)
    volume_df = volume_df.fillna(method = 'ffill')
    volume_df = volume_df.fillna(method = 'bfill')
    
    sma, p_sma = p_sma_ratio(prices_df, lookback = 30)
    
    sma_norm = sma/sma.iloc[0]
    sd = dt.datetime(2008,6,1)
    ed = dt.datetime(2009,1,1)
    #plot the Price/SMA ratio
    gridspec.GridSpec(15,1)    
    plt.subplot2grid((16,1), (0,0), colspan=1, rowspan=12)
    plt.plot(prices_df.index, prices_df['JPM']/prices_df['JPM'][0],label = "Price")
    plt.plot(sma.index, sma_norm, label = "SMA")
    plt.plot(p_sma.index, p_sma, label = "Price/SMA ratio")
    plt.legend(["Price","SMA","Price/SMA ratio"],loc = 'lower left')
    plt.title("Price/SMA ratio")
    plt.ylabel("Normalized Price")
    plt.xlim(sd, ed)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.ylim(0.3, 1.3)
    plt.grid(color='gray', linestyle='--', linewidth=1)
    plt.subplot2grid((16,1), (14,0),colspan=1, rowspan=3)
    y= np.zeros((volume_df.shape[0],))
    plt.plot(volume_df.index, volume_df['JPM']/1000000, volume_df.index, y, color = "r")
    plt.fill_between(volume_df.index, volume_df['JPM']/1000000, y, color='r', alpha = 0.4)
    plt.ylabel("Volume (1M)")
    plt.xlim(sd, ed)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.ylim(0.0, 210)
    plt.grid(color='gray', linestyle='--', linewidth=1)
    plt.savefig('sma_indicator.png', dpi=300)
    
    
    #plot the stochastic indicator
    K, D = stochastic_K_D(prices_df, lookback = 30)
    gridspec.GridSpec(15,1)    
    plt.subplot2grid((16,1), (0,0), colspan=1, rowspan=12)
    plt.plot(K.index, K['JPM'])
    plt.plot(D.index, D['JPM'])
    overbought = np.full((len(K.index),), 80.0)
    oversold = np.full((len(K.index),), 20.0)
    plt.fill_between(K.index, overbought, oversold, color='grey', alpha=0.25)
    plt.legend(['%K','%D'],ncol = 1, loc =1)
    plt.title("Stochastic Oscillator")
    plt.xlim(sd, ed)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.grid(color='gray', linestyle='--', linewidth=1)
    plt.ylabel("%K, %D")

    plt.subplot2grid((16,1), (14,0),colspan=1, rowspan=3)
    y= np.zeros((volume_df.shape[0],))
    plt.plot(volume_df.index, volume_df['JPM']/1000000, volume_df.index, y, color = "r")
    plt.fill_between(volume_df.index, volume_df['JPM']/1000000, y, color='r', alpha =0.4)
    plt.ylabel("Volume (1M)")
    plt.xlim(sd, ed)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.ylim(0.0, 210)
    plt.grid(color='gray', linestyle='--', linewidth=1)
    plt.savefig('stochastic_indicator.png', dpi=300)
    
   
    #plot the Bollinger Bands indicator
    BB, upperband, lowerband = bband(prices_df, lookback = 30)
    gridspec.GridSpec(15,1)    
    plt.subplot2grid((16,1), (0,0), colspan=1, rowspan=10)
    plt.grid(color='gray', linestyle='--', linewidth=1, alpha = 0.4)
    plt.plot(sma.index, sma/sma.iloc[0], label = "SMA")
    plt.plot(prices_df.index, prices_df['JPM']/prices_df['JPM'][0])
    plt.fill_between(upperband.index, upperband['JPM'], lowerband['JPM'], color='grey', alpha=0.20)
    plt.title("Bollinger Band Indicator")
    plt.legend(['SMA', 'price','Bollinger Band'],loc=3)
    plt.ylabel("Normalized Price")
    plt.xlim(sd, ed)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.subplot2grid((16,1), (11,0),colspan=1, rowspan=5)
    
    plt.plot(BB.index, BB, color = "red")
    plt.ylabel("BB Range")
    plt.xlim(sd, ed)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.grid(color='grey', linestyle='--', linewidth=1, alpha=0.40)
    plt.legend(['BB indicator'],loc="upper right")
    plt.savefig('BB_indicator.png', dpi=300)
    
    
if __name__ == "__main__":
    main()