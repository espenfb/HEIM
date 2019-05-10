# -*- coding: utf-8 -*-
"""
Created on Fri May  3 11:37:54 2019

@author: espenfb
"""

import pandas as pd
import geopandas as gpd


#load = pd.read_csv('ERCOT-load-2017.csv', index_col = 0, parse_dates = [0])
load = pd.read_csv('zone_load_2015.csv', index_col = 0, parse_dates = [0], 
                   skipinitialspace = True)

zip_county = pd.read_csv('..\\geo\\zip_city_county_texas.csv',
                         skipinitialspace = True)


pop = pd.read_csv('..\\geo\\2017_txpopest_county.csv')



zip_zone = pd.read_csv('zip_weather_zone.csv', skipinitialspace = True)



zip_info = pd.merge(zip_zone, zip_county, on = ['Zip Code'], how = 'inner' )

zip_info = pd.merge(zip_info, pop, on = ['County'], how = 'inner' )


county_pop = zip_info.groupby(['Weather Zone Name', 'County']).agg({'jan1_2018_pop_est':'sum'})


tot_pop_zone = county_pop.sum(level = 0)

for c in county_pop.index.levels[0]:
    county_pop.loc[c, 'load_share'] = (county_pop.loc[c]/tot_pop_zone.loc[c])['jan1_2018_pop_est'].tolist()
    
    
    



buses = pd.read_excel('..\\grid\\13_Bus_Case.xlsx', index_col = 0)

tx = gpd.read_file('..\\geo\\Texas_County_Boundaries_Detailed\\Texas_County_Boundaries_Detailed.shp')

tx_point = pd.read_csv('..\\geo\\Texas_Counties_Centroid_Map.csv')

import numpy as np
for b in buses.index:
    lat = buses.Lat.loc[b]
    lon = buses.Lon.loc[b]
    dist = np.sqrt((lon - tx_point['X (Long)'])**2 + (lat - tx_point['Y (Lat)'])**2)
    if 'min_dist' in tx_point.columns:
        min_bool = dist <= tx_point['min_dist']
        tx_point.loc[min_bool, 'min_dist'] = dist.loc[min_bool]
        tx_point.loc[min_bool, 'closest_bus'] = b
    else:
        tx_point.loc[:,'min_dist'] = dist
        tx_point.loc[:,'closest_bus'] = b

tx_point.rename(columns = {'CNTY_NM':'County'}, inplace = True)

county_pop_df = county_pop.reset_index(level = ['Weather Zone Name', 'County'])

bus_info = pd.merge(county_pop_df, tx_point, on = ['County'], how = 'inner')

bus_county = bus_info.groupby(['closest_bus','County']).agg({'jan1_2018_pop_est':'sum',
                             'Weather Zone Name':'sum','load_share':'sum'})
bus_county.to_excel('bus_county_load.xlsx')
    
bus_zone = bus_info.groupby(['closest_bus','Weather Zone Name']).agg({'load_share':'sum'})

bus_zone.to_excel('bus_load.xlsx')

load.columns = load.columns.str.title()
load.rename(columns = {'Far_West': 'Far West', 'North_C': 'North Central',
                    'South_C' : 'South Central', 'Southern': 'South'},
    inplace = True)
#load.rename(columns = {'Fwest': 'Far West', 'Ncent': 'North Central',
#                    'Scent' : 'South Central'},inplace = True)
    
bus_loads = pd.DataFrame(columns = bus_zone.index.levels[0])
for i in bus_zone.index.levels[0]:
    bus_load = pd.Series()
    for j in bus_zone.loc[i].index:
        if bus_load.empty:
            bus_load = bus_zone.loc[i].loc[j]['load_share']*load[j]
        else:
            bus_load += bus_zone.loc[i].loc[j]['load_share']*load[j]
    bus_loads[i] = bus_load 


import matplotlib.pyplot as plt

plt.figure('Bus profiles')
ax = plt.gca()
bus_loads.plot(ax = ax)

bus_loads.to_csv('bus_loads.csv')

plt.figure('Tot comp')
ax2 = plt.gca()
bus_loads.sum(axis = 1).plot(ax = ax2)
load['Ercot'].plot(ax = ax2)

# Check the number of times the total series differ with more than 0.1%
sum((load['Ercot'] <= 0.999*bus_loads.sum(axis = 1)) & (load['Ercot'] >= 1.001*bus_loads.sum(axis = 1)))