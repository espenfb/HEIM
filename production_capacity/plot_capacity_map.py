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

#for fuel in cap_gdf.FUEL.unique():
#    fig = plt.figure(fuel)
#    ax = plt.gca()
#    texas.plot(ax = ax)
#    #centroid_gdf.plot(ax = ax, color = 'lightgrey')
#    buses_gdf.plot(ax = ax, color = 'r')
#    cap_gdf[cap_gdf.FUEL == fuel].plot(ax = ax, color = 'y')
#    ax.set_title(fuel)
#    ax.figure.set_label(fuel)
#    fig.savefig(fuel + '_loc_texas.pdf')



# math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
import numpy as np
for b in buses.index:
    
    lat = buses.Lat.iloc[b]
    lon = buses.Lon.iloc[b]
    dist = list(np.sqrt((lat - cap['X (Lat)'])**2 + (lon - cap['Y (Long)'])**2))
    if not 'Min_dist' in cap.keys():
        cap['Min_dist'] = dist
    else:
        indx = dist < cap['Min_dist']
        cap.loc[indx.tolist(),:][ 'Min_dist'] = dist[indx]
