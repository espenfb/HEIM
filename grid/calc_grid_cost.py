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
                                 np.cos(from_lat)*np.cos(to_lat)*np.power(np.sin((to_lon - from_lon)/2), 2)))
    

inv_cost_mile_kW = 0.93 # $/(mile*kW)
terrain_factor = 1.3

mile2km = 1.6
kW2MW = 0.001

inv_cost_km = (inv_cost_mile_kW*terrain_factor)/(kW2MW*mile2km)
inv_cost_mile = (inv_cost_mile_kW*terrain_factor)/(kW2MW)

irr = 0.06 
lifetime = 40.0 # years

epsilon = irr/(1-(1+irr)**(-lifetime))

ann_cost_km = inv_cost_km*epsilon
ann_cost_mile = inv_cost_mile*epsilon

#line.loc[:,'Cost'] = line.loc[:,'Distance']*ann_cost_km 
line.loc[:,'Cost'] = (line.loc[:,'Length']*ann_cost_mile)

exists = line.index[line.loc[:,'Type'] == 'Existing']


line.loc[exists,'Cost'] = 0.0

line.loc[:,'Cost'] = line.loc[:,'Cost'].round(0)

line.to_csv('line_data_cost.csv', index = False)