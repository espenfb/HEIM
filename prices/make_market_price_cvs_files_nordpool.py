# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:49:33 2019

@author: espenfb
"""

import pandas as pd
import datetime
from tqdm import tqdm

folder = 'Nordpool\\'

#filename = 'RTM_2013.xlsx'
years = [2013, 2014, 2015, 2016, 2017, 2018, 2019]
EUR2USD = {2013: 1.2770, 2014: 1.2755, 2015: 1.0672, 2016: 1.0638,
           2017: 1.0834, 2018: 1.1792, 2019: 1.14 }

for y in tqdm(years):

    #filename = 'elspot-prices_' + str(y) +'_hourly_eur.xlsx'
    filename = 'regulating-prices_' + str(y) +'_hourly_eur.xlsx'
    skip_rows = [0,1]
    header = [0,1]
    #header = 0
    #levels = None
    levels = [0]
    
    market_prices = pd.read_excel(folder + filename,
                                  skiprows = skip_rows,
                                  header = header,
                                  index_col = 0,
                                  parse_dates = [[0,1]])
    
    
    
    def parseIndexToDatetime(index):
        
        new_index = []
        for n, i in enumerate(index):
            try:
                year = int(i[:4])
                month = int(i[5:7])
                day = int(i[8:10])
                hour = int(i[20:22])
            except:
                print(i, n)
                continue
    
            new_index.append(datetime.datetime(year, month, day, hour))
            
        return pd.DatetimeIndex(new_index)
    
    
    market_prices.index = parseIndexToDatetime(market_prices.index)
    market_prices = market_prices#*EUR2USD[y]
    market_prices.columns.set_names('Regulating prices in USD/MWh',
                                    level = levels,
                                    inplace = True) 
    
    year = market_prices.index[0].year
    market_prices.to_csv('Nordpool/REG_' + str(year) + '_EUR.csv')