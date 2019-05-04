# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 19:49:17 2019

@author: espenfb
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt


cap = pd.read_excel('..\\production_capacity\\CapacityDemandandReservesReport-Dec2018.xlsx',
                    sheet_name = 'WinterCapacities', skiprows = [0,2], index_col = 0)

cap.dropna(subset = ['COUNTY'], inplace = True)
cap.dropna(subset = ['FUEL'], inplace = True)
cap.index = [int(i) for i in cap.index]
cap.COUNTY = cap.COUNTY.str.title()
cap.COUNTY.replace('Ft. Bend', 'Fort Bend', inplace = True)
cap.COUNTY.replace('Mclennan', 'McLennan', inplace = True)
cap.COUNTY.replace('Mcculloch', 'McCulloch', inplace = True)

texas = gpd.read_file('..\\geo\\Texas_County_Boundaries_line\\Texas_County_Boundaries_line.shp')

centroid = pd.read_csv('..\\geo\\Texas_Counties_Centroid_Map.csv')
centroid['Coordinates'] = list(zip(centroid['X (Lat)'], centroid['Y (Long)']))
centroid['Coordinates'] = centroid['Coordinates'].apply(Point)
centroid_gdf = gpd.GeoDataFrame(centroid, geometry = 'Coordinates')

buses = pd.read_excel('..\\grid\\13_Bus_Case.xlsx', sheet_name = 'Bus', index_col = 0)
buses['Coordinates'] = list(zip(buses['Lon'], buses['Lat']))

buses['Coordinates'] = buses['Coordinates'].apply(Point)

buses_gdf = gpd.GeoDataFrame(buses, geometry = 'Coordinates')


#ax = texas.plot()
#centroid_gdf.plot(ax = ax, color = 'lightgrey')
#buses_gdf.plot(ax = ax, color = 'r')

county_coord = centroid.loc[:,['Coordinates','CNTY_NM', 'X (Lat)','Y (Long)']]
county_coord.rename(columns={'CNTY_NM':'COUNTY'}, inplace = True)

cap = pd.merge(cap, county_coord, on = ['COUNTY'], how = 'inner' )
cap_gdf = gpd.GeoDataFrame(cap, geometry = 'Coordinates')

for fuel in cap_gdf.FUEL.unique():
    fig = plt.figure(fuel)
    ax = plt.gca()
    texas.plot(ax = ax)
    #centroid_gdf.plot(ax = ax, color = 'lightgrey')
    buses_gdf.plot(ax = ax, color = 'r')
    cap_gdf[cap_gdf.FUEL == fuel].plot(ax = ax, color = 'y')
    ax.set_title(fuel)
    ax.figure.set_label(fuel)
    fig.savefig(fuel + '_loc_texas.pdf')



# math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
import numpy as np
for b in buses.index:
    print(b)
    lat = buses.Lat.loc[b]
    lon = buses.Lon.loc[b]
    dist = np.sqrt((lon - cap['X (Lat)'])**2 + (lat - cap['Y (Long)'])**2)
    print(sum(dist))
    if 'min_dist' in cap.columns:
        min_bool = dist <= cap['min_dist']
        print(sum(min_bool))
        cap.loc[min_bool, 'min_dist'] = dist.loc[min_bool]
        cap.loc[min_bool, 'closest_bus'] = b
    else:
        print(b)
        cap.loc[:,'min_dist'] = dist
        cap.loc[:,'closest_bus'] = b
        
        
sum_df = cap.groupby(['COUNTY','FUEL']).agg({'2028/2029': 'sum'})
sum_df = cap.groupby(['closest_bus','COUNTY','FUEL']).agg({'2028/2029': 'sum'})

sum_df = cap.groupby(['closest_bus','FUEL']).agg({'2028/2029': 'sum'})

cap_df = sum_df.unstack()
cap_df.to_csv('Installed_cap.csv')
cap_df.plot(kind = 'bar')