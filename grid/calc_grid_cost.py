# -*- coding: utf-8 -*-
"""
Created on Wed May  8 19:46:07 2019

@author: espenfb
"""

import pandas as pd
import numpy as np

bus = pd.read_csv('bus_data.csv', index_col = 0, skipinitialspace = True)

line = pd.read_csv('line_data.csv', skipinitialspace = True)

R = 6378.1 # radius of earth in km

for i in line.index:
    from_bus = line.iloc[i].From
    to_bus = line.iloc[i].To
    
    from_lon = bus.loc[from_bus].Lon
    from_lat = bus.loc[from_bus].Lat
    
    to_lon = bus.loc[to_bus].Lon
    to_lat = bus.loc[to_bus].Lat
    
    
    
    line.loc[i,'Distance'] = 2*R*np.arcsin(np.sqrt(np.power(np.sin((to_lat - from_lat)/2), 2) + 
                                 np.cos(from_lat)*np.cos(to_lat))*np.power(np.sin((to_lon - from_lon)/2), 2))
    

inv_cost_mile = 1 # M$/mile

mile2km = 1/1.6

inv_cost_km = inv_cost_mile*mile2km

irr = 0.06 
lifetime = 40.0 # years

epsilon = irr/(1-(1+irr)**(-lifetime))

ann_cost_km = inv_cost_km*epsilon

line.loc[:,'Cost'] = line.loc[:,'Distance']*ann_cost_km 

exists = line.index[line.loc[:,'Type'] == 'Existing']


line.loc[exists,'Cost'] = 0.0

line.loc[:,'Cost'] = line.loc[:,'Cost'].round(0)

line.to_csv('line_data_cost.csv', index = False)