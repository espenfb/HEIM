# -*- coding: utf-8 -*-
"""
Created on Sun May  5 20:09:19 2019

@author: espenfb
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt



centroid = pd.read_csv('..\\geo\\Texas_Counties_Centroid_Map.csv')
centroid['Coordinates'] = list(zip(centroid['X (Long)'], centroid['Y (Lat)'])) # Long and Lat are wrong here ....
centroid['Coordinates'] = centroid['Coordinates'].apply(Point)
county = gpd.GeoDataFrame(centroid, geometry = 'Coordinates')

county.rename(columns = {'CNTY_NM' : 'County'}, inplace = True)

buses = pd.read_excel('..\\grid\\13_Bus_Case.xlsx', sheet_name = 'Bus', index_col = 0)
buses['Coordinates'] = list(zip(buses['Lon'], buses['Lat']))

buses['Coordinates'] = buses['Coordinates'].apply(Point)

buses_gdf = gpd.GeoDataFrame(buses, geometry = 'Coordinates')



import numpy as np
for b in buses.index:
    lat = buses.Lat.loc[b]
    lon = buses.Lon.loc[b]
    dist = np.sqrt((lon - county['X (Long)'])**2 + (lat - county['Y (Lat)'])**2) # Long and Lat are wrong here ....
    if 'Dist' in county.columns:
        min_bool = dist <= county['Dist']
        county.loc[min_bool, 'Dist'] = dist.loc[min_bool]
        county.loc[min_bool, 'Bus'] = b
    else:
        county.loc[:,'Dist'] = dist
        county.loc[:,'Bus'] = b
        
        
bus_county = county.groupby(['Bus','County']).agg({'Dist':'sum'})   

bus_county.reset_index(level=['Bus','County'], inplace=True)

bus_county.drop(columns = ['Dist'], axis = 1, inplace = True)
bus_county.to_csv('bus_county.csv')
