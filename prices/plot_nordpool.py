# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:32:33 2019

@author: espenfb
"""

import pandas as pd
import numpy as np

folder = 'Nordpool\\'

NO_zone_map = {'NO1': 'Oslo', 'NO2':'Kr.sand', 'NO3':'Tr.heim', 'NO4':'Tromsø',
               'NO5': 'Bergen'}

NO_zone_map = {v: k for k, v in NO_zone_map.items()}

years = np.arange(2013,2019)

spot = pd.DataFrame()
reg = pd.DataFrame()

for y in years:
    spot_year = pd.read_csv(folder + 'SPOT_'+ str(y) + '_EUR.csv', index_col = 0,
                            parse_dates = [0], encoding = 'latin1')
    
    new_col = []
    for i in spot_year.columns:
        if i in NO_zone_map.keys():
            new_col.append(NO_zone_map[i])
        else:
            new_col.append(i)
    spot_year.columns = new_col
    
    if spot.empty:
        spot = spot_year
    else:
        spot = pd.concat([spot, spot_year], sort=True)
        
        
        
    reg_year = pd.read_csv(folder + 'REG_'+ str(y) + '.csv', index_col = 0,
                            parse_dates = [0], header = [0,1], encoding = 'latin1')
    if reg.empty:
        reg = reg_year
    else:
        reg = pd.concat([reg, reg_year])
        
premium = pd.DataFrame(columns = reg.columns)
for z in reg.columns.levels[0]:
    for i in reg.columns.levels[1]:
        if i == 'Up':
            premium.loc[:,(z,i)] = reg.loc[:,(z,i)] - spot.loc[:,z]
        elif i == 'Down':
            premium.loc[:,(z,i)] = spot.loc[:,z] - reg.loc[:,(z,i)]
            
    
    
     
