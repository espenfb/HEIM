# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:46:49 2019

@author: espenfb
"""

import pandas as pd

import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

texas = gpd.read_file('..\\geo\\Texas_County_Boundaries_line\\Texas_County_Boundaries_line.shp')

centroid = pd.read_csv('..\\geo\\Texas_Counties_Centroid_Map.csv')
centroid['Coordinates'] = list(zip(centroid['X (Lat)'], centroid['Y (Long)']))
centroid['Coordinates'] = centroid['Coordinates'].apply(Point)
centroid_gdf = gpd.GeoDataFrame(centroid, geometry = 'Coordinates')

buses = pd.read_excel('..\\grid\\13_Bus_Case.xlsx', sheet_name = 'Bus', index_col = 0)
buses['Coordinates'] = list(zip(buses['Lon'], buses['Lat']))

buses['Coordinates'] = buses['Coordinates'].apply(Point)

buses_gdf = gpd.GeoDataFrame(buses, geometry = 'Coordinates')

plt.figure('Wind and Solar')
ax = plt.gca()
texas.plot(ax = ax)
#centroid_gdf.plot(ax = ax, color = 'lightgrey')
buses_gdf.plot(ax = ax, color = 'r')
# We restrict to South America.
#ax = world[world.continent == 'North America'].plot(
#    color='white', edgecolor='black')

# We can now plot our GeoDataFrame.
#gdf.plot(ax=ax, color='red')
#
#plt.show()


wind = pd.read_csv('..\\energy_profiles\\wind_cite_id.csv', skiprows = [0], skipinitialspace = True)
wind['Coordinates'] = list(zip(wind['LONGITUDE'], wind['LATITUDE']))
wind['Coordinates'] = wind['Coordinates'].apply(Point)
wind_gdf = gpd.GeoDataFrame(wind, geometry = 'Coordinates')
wind_gdf.plot(ax = ax, color = 'g')


county_coord = centroid.loc[:,['Coordinates','CNTY_NM']]
county_coord.rename(columns={'CNTY_NM':'County'}, inplace = True)

solar = pd.read_csv('..\\energy_profiles\\solar_cite_id.csv')

solar = pd.merge(solar, county_coord, on = ['County'], how = 'inner' )
solar_gdf = gpd.GeoDataFrame(solar, geometry = 'Coordinates')
solar_gdf.plot(ax = ax, color = 'y')


#ax.legend(['county boundaries','county centroid','buses', 'wind','solar'])
ax.legend(['county boundaries','buses', 'wind','solar'])
ax.set_title('Wind and Solar')
plt.savefig('wind_and_solar_loc_texas.pdf')





