# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:46:34 2019

@author: espenfb
"""

import pandas as pd
from shapely.geometry import Point
import geopandas as gpd

filename = 'ERCOT_onshore_2015.CSV'

wind = pd.read_csv(filename, header = 0, index_col = 0, parse_dates = [[0,1]])

wind_id_cap = pd.DataFrame(columns = ['ID','CAP'], dtype = np.float64)
for n, i in enumerate(wind.columns):
    w_id = int(i[5:10])
    w_cap = float(i[20:])
    wind_id_cap.loc[n,'ID'] = w_id
    wind_id_cap.loc[n,'CAP'] = w_cap
    
wind.columns = wind_id_cap['ID']

wind = wind[wind.index.year == 2015]

wind_cite_id = pd.read_csv('..\\wind_cite_id_all.csv',
                           skipinitialspace = True, skiprows = [0])

wind_cite_id.rename(columns = {'SELECTED SITESSITE_ID' : 'ID'}, inplace = True)

wind_info = pd.merge(wind_id_cap,wind_cite_id, on = ['ID'], how='inner')

buses = pd.read_excel('..\\..\\grid\\13_Bus_Case.xlsx', sheet_name = 'Bus', index_col = 0)
buses['Coordinates'] = list(zip(buses['Lon'], buses['Lat']))

buses['Coordinates'] = buses['Coordinates'].apply(Point)

buses_gdf = gpd.GeoDataFrame(buses, geometry = 'Coordinates')

import numpy as np
for b in buses.index:
    lat = buses.Lat.loc[b]
    lon = buses.Lon.loc[b]
    dist = np.sqrt((lon - wind_info['LONGITUDE'])**2 + (lat - wind_info['LATITUDE'])**2)
    if 'min_dist' in wind_info.columns:
        min_bool = dist <= wind_info['min_dist']
        wind_info.loc[min_bool, 'min_dist'] = dist.loc[min_bool]
        wind_info.loc[min_bool, 'closest_bus'] = b
    else:
        wind_info.loc[:,'min_dist'] = dist
        wind_info.loc[:,'closest_bus'] = b
        


bus  = wind_info.groupby(['closest_bus','PLANT TYPE','ID']).agg({'CAP':'sum'})

bus_type = wind_info.groupby(['closest_bus','PLANT TYPE']).agg({'CAP':'sum'})
bus_type.unstack().plot(kind = 'bar')
bus_type = bus_type.reset_index(level = [0,1])
bus_type.rename(columns = {'closest_bus' : 'Bus', 'PLANT TYPE':'Plant type',
                           'CAP':'Cap'}, inplace = True)
bus_type.to_csv('wind_farms.csv',index = False)
bus_tot = bus.sum(level = 0)

wind_profiles = pd.DataFrame()
for i in bus.index.levels[0]:
    wind_profile_type = pd.DataFrame()
    for t in bus.loc[i].index.get_level_values(level = 0).unique():
        for j in bus.loc[i,t].index:
            if not t in wind_profile_type.columns:
                wind_profile_type[t] = wind[j]
            else:
                wind_profile_type[t] += wind[j]
    wind_profile_type.columns = pd.MultiIndex.from_product([[i],wind_profile_type.columns])
    wind_profiles = pd.concat([wind_profiles,wind_profile_type], axis = 1)

wind_profiles.plot()

wind_profiles.to_csv('wind_profiles_bus_all_types.csv')

inst_cap = pd.read_csv('..\\..\\production_capacity\\Installed_cap.csv',
                       index_col = 0, skiprows = [0,2])


