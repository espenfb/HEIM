# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:49:33 2019

@author: espenfb
"""

import pandas as pd
import datetime

folder = 'Nordpool\\'

filename = 'RTM_2013.xlsx'


market_prices_xslx = pd.read_excel(folder + filename, header = 0, index_col = 0,
                                 parse_dates = [[0,1]],
                                 sheetname = None)



def parseIndexToDatetime(index, interval = pd.Series([])):
    
    new_index = []
    for n, i in enumerate(index):
        try:
            month = int(i[:2])
            day = int(i[3:5])
            year = int(i[6:10])
            hour = int(i[11:13]) - 1
            if interval.empty:
                minutes = (interval[n]-1)*15
            else:
                minutes = 0
        except:
            print(i, n)
            continue

        new_index.append(datetime.datetime(year, month, day, hour, minutes))
        
    return pd.DatetimeIndex(new_index)

#market_prices_xslx.index = parseIndexToDatetime(market_prices_xslx.index)



def createColumnsForPoints(market_prices_xslx):
    
    new_df = pd.DataFrame()
    
    for sheet in market_prices_xslx.keys():
        
        if 'Delivery Interval' in market_prices_xslx[sheet].columns:
            drop_indx = market_prices_xslx[sheet]['Settlement Point Type'] != 'LZEW'
            new_indx = market_prices_xslx[sheet].index[drop_indx]
            market_prices_xslx[sheet] = market_prices_xslx[sheet][drop_indx]
            market_prices_xslx[sheet].index = new_indx
            sub_hour_interval = market_prices_xslx[sheet]['Delivery Interval']
        else:
            sub_hour_interval = pd.Series([])
        
        market_prices_xslx[sheet].index = parseIndexToDatetime(market_prices_xslx[sheet].index, 
                          interval = sub_hour_interval)
    
        for col in range(market_prices_xslx[sheet].shape[0]):
            
            if 'Settlement Point' in market_prices_xslx[sheet]:
                point = market_prices_xslx[sheet].iloc[col]['Settlement Point']
            elif  'Settlement Point Name' in market_prices_xslx[sheet]:
                point = market_prices_xslx[sheet].iloc[col]['Settlement Point Name']
                
            price = market_prices_xslx[sheet].iloc[col]['Settlement Point Price']
            date = market_prices_xslx[sheet].iloc[col].name
            
            new_df.loc[date, point] = price
            
    return new_df

market_prices = createColumnsForPoints(market_prices_xslx)


year = market_prices.index[0].year
market_prices.to_csv('../Prices/RTM_' + str(year) + '.csv')