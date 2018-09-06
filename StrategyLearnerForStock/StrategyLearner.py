"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
Student: Si Yan
userID: syan62
"""

import datetime as dt
import pandas as pd
import numpy as np
import util as ut
import random
from marketsimcode import compute_portvals
from indicators import p_sma_ratio, stochastic_K_D, bband
import QLearner as ql

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.sv = 0
        self.netHolding = 0;
        self.learner = ql.QLearner(num_states=800,\
            num_actions = 3, \
            alpha = 0.2, \
            gamma = 0.9, \
            rar = 0.5, \
            radr = 0.95, \
            dyna = 0, \
            verbose = self.verbose) #initialize the learner

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 100000):

        # add your code to do learning here
        self.sv = sv
        #Read in the stock price
        if type(symbol) is str:
            symbols = list()
            symbols.append(symbol)
        else:
            symbols = symbol
        prices_df = ut.get_data(symbols, pd.date_range(sd, ed), addSPY=True, colname = 'Adj Close')
        #prices_df['cash'] = 1.0
        #prices_df = prices_df.drop(['SPY'], axis = 1)
        prices_df = prices_df.fillna(method = 'ffill')
        prices_df = prices_df.fillna(method = 'bfill')
        prices_df = prices_df.drop(['SPY'], axis = 1)

        lookback_days = 15

        #Create an empty dataframe df_trades
        df_trades = pd.DataFrame(data = 0, index = prices_df.index, columns = symbols)
        #print df_trades.head(20)
        sma, p_sma = p_sma_ratio(prices_df, lookback = lookback_days)
        K, D = stochastic_K_D(prices_df, lookback = lookback_days)
        BB, upperband, lowerband = bband(prices_df, lookback = lookback_days)

        #discretize the indicators
        p_sma_bins = pd.to_numeric(pd.cut(p_sma[symbol], 10, labels = range(0,10)))
        K_bins = pd.to_numeric(pd.cut(K[symbol], 8, labels = range(0,8)))
        BB_bins = pd.to_numeric(pd.cut(BB[symbol], 10, labels = range(0,10)))

        states = 100*K_bins + 10*p_sma_bins+ BB_bins
        #print type(states[0])
        #Instantiate a Q-learner

        converge = False
        count = 0
        #portvals_prev = [sv, sv, sv, sv, sv, sv, sv, sv, sv, sv]
        portvals_prev = [sv, sv, sv, sv, sv]
        #df_tmp = df_trades
        while (not converge and count < 500):
            action = self.learner.querysetstate(states[0])

            if (action == 0): #long
                self.netHolding = 1000
            elif (action == 1): #Nothing
                self.netHolding = 0
            else: #short
                self.netHolding = -1000

            #print df_trades.loc['2008-01-02':'2008-01-31']
    #        print states.index
    #        print states.shape[0]
    #        print prices_df.loc[states.index[0]]

            for i in range(1, states.shape[0]):
                day = states.index[i]
                day_before = states.index[i-1]
                if (action == 0):
                    impact_sign = 1
                elif (action == 1):
                    impact_sign = 0
                else:
                    impact_sign = -1

                #r = (prices_df[symbol].loc[day] - prices_df[symbol].loc[day_before] * \
                #   (1+impact_sign * self.impact)) * self.netHolding / self.sv
                r = self.netHolding * (prices_df[symbol].loc[day]/prices_df[symbol].loc[day_before] - 1 - self.impact*impact_sign)
#                if self.netHolding != 0:    
#                    r = prices_df[symbol].loc[day]/prices_df[symbol].loc[day_before] - 1 - self.impact
#                else:
#                    r = 0
                #r = ((prices_df[symbol].loc[day] - prices_df[symbol].loc[day_before]) * self.netHolding  \
                #    - impact_sign * self.impact * df_trades.loc[day_before] * prices_df[symbol].loc[day_before])/self.sv
                action = self.learner.query(states[day], r)

                if (action == 0): #long
                    if self.netHolding == -1000:
                        df_trades.loc[day] = 2000
                    elif self.netHolding == 0:
                        df_trades.loc[day] = 1000
                elif (action == 2): #Short
                    if self.netHolding == 1000:
                        df_trades.loc[day] = -2000
                    elif self.netHolding == 0:
                        df_trades.loc[day] = -1000

                self.netHolding += df_trades.loc[day].values
    #            print prices_df.loc[day]
    #            print df_trades.loc[day]
    #            print self.netHolding
    #            print ""

            portvals, cr, stdev, mean = compute_portvals(df_trades, start_val = sv, commission=0, impact=self.impact)
            #diff_portvals = abs(portvals[-1]- portvals_prev)
            diff_portvals = abs(portvals[-1]- np.mean(portvals_prev))

            #print portvals_prev

            if (count >= 10 and diff_portvals < 50.0):
                converge = True

            else:
                count += 1
                portvals_prev.pop(0)
                portvals_prev.append(portvals[-1])
        print count
#        print portvals[-1]

        # example usage of the old backward compatible util function
#        syms=[symbol]
#        dates = pd.date_range(sd, ed)
#        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
#        prices = prices_all[syms]  # only portfolio symbols
#        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
#        if self.verbose: print prices
#
#        # example use with new colname
#        volume_all = ut.get_data(syms, dates, colname = "Volume")  # automatically adds SPY
#        volume = volume_all[syms]  # only portfolio symbols
#        volume_SPY = volume_all['SPY']  # only SPY, for comparison later
#        if self.verbose: print volume

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "JPM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):


        #Read in the stock price
        if type(symbol) is str:
            symbols = list()
            symbols.append(symbol)
        else:
            symbols = symbol
        prices_df = ut.get_data(symbols, pd.date_range(sd, ed), addSPY=True, colname = 'Adj Close')
        #prices_df['cash'] = 1.0
        #prices_df = prices_df.drop(['SPY'], axis = 1)
        prices_df = prices_df.fillna(method = 'ffill')
        prices_df = prices_df.fillna(method = 'bfill')
        prices_df = prices_df.drop(['SPY'], axis = 1)

        lookback_days = 15

        #Create an empty dataframe df_trades
        df_trades = pd.DataFrame(data = 0, index = prices_df.index, columns = symbols)
        #print df_trades.head(20)
        sma, p_sma = p_sma_ratio(prices_df, lookback = lookback_days)
        K, D = stochastic_K_D(prices_df, lookback = lookback_days)
        BB, upperband, lowerband = bband(prices_df, lookback = lookback_days)

        #discretize the indicators
        p_sma_bins = pd.to_numeric(pd.cut(p_sma[symbol], 10, labels = range(0,10)))
        K_bins = pd.to_numeric(pd.cut(K[symbol], 8, labels = range(0,8)))
        BB_bins = pd.to_numeric(pd.cut(BB[symbol], 10, labels = range(0,10)))

        states = 100*K_bins + 10* p_sma_bins + BB_bins
        netHolding = 0

        for i in range(1, states.shape[0]):
            day = states.index[i]
            #day_before = states.index[i-1]

            #r = (prices_df[symbol].loc[day] - prices_df[symbol].loc[day_before])* netHolding / sv
            action = self.learner.querysetstate(states[day])

            if (action == 0): #long
                if netHolding == -1000:
                    df_trades.loc[day] = 2000
                elif netHolding == 0:
                    df_trades.loc[day] = 1000
            elif (action == 2): #Short
                if netHolding == 1000:
                    df_trades.loc[day] = -2000
                elif netHolding == 0:
                    df_trades.loc[day] = -1000

            netHolding += df_trades.loc[day].values

        #portvals, cr, stdev, mean = compute_portvals(df_trades, start_val = sv, commission=0, impact=self.impact)
        #    print states[day]
        # here we build a fake set of trades
        # your code should return the same sort of data
#        dates = pd.date_range(sd, ed)
#        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
#        trades = prices_all[[symbol,]]  # only portfolio symbols
#        trades_SPY = prices_all['SPY']  # only SPY, for comparison later
#        trades.values[:,:] = 0 # set them all to nothing
#        trades.values[0,:] = 1000 # add a BUY at the start
#        trades.values[40,:] = -1000 # add a SELL
#        trades.values[41,:] = 1000 # add a BUY
#        trades.values[60,:] = -2000 # go short from long
#        trades.values[61,:] = 2000 # go long from short
#        trades.values[-1,:] = -1000 #exit on the last day
#        if self.verbose: print type(trades) # it better be a DataFrame!
#        if self.verbose: print trades
#        if self.verbose: print prices_all
        return df_trades

def main():
    import StrategyLearner as sl
    learner = sl.StrategyLearner(verbose = False, impact = 0.000) # constructor
    learner.addEvidence(symbol="UNH",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=100000)
    df_trades = learner.testPolicy(symbol="UNH",sd=dt.datetime(2010,1,1),ed=dt.datetime(2011,12,31),sv=100000)
    portvals, cr, stdev, mean = compute_portvals(df_trades, start_val = 100000, commission=0, impact=0)
    #print df_trades
    print "portvals, cr, stdev, mean: {},{},{},{}".format(portvals[-1], cr, stdev, mean)
# sl = StrategyLearner()
#    sv = 100000
#    sl.addEvidence(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = sv)

if __name__=="__main__":
    print "One does not simply think up a strategy"
    main()
