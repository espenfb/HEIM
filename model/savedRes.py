# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:22:31 2019

@author: espenfb
"""

import os
import pandas as pd
import numpy as np

import geopandas as gpd
#import geoplot as gplt
#import geoplot.crs as gcrs
import matplotlib.pyplot as plt
#import cartopy
from shapely.geometry import Point, LineString
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm
import matplotlib as mpl
        
        
class savedRes(object):
    
    def __init__(self, folder, data = None):
    
        for file in os.listdir(folder):
            if file[:-4] == 'investments':
                df = pd.read_csv(folder + file, index_col = 0, header = [0]).replace(np.NaN,0)
            else:
                df = pd.read_csv(folder + file, index_col = 0, header = [0,1]).replace(np.NaN,0)
                    
            setattr(self, file[:-4], df)
            
        self.data = data
            
        self.type_identifier = {'HS': 'H2_Storage', 'B': 'Biomass', 'S': 'Solar',
                           'N': 'Nuclear', 'E': 'Elec', 'C': 'Coal',
                           'G': 'Gas', 'W': 'Wind'}
        
        
    def invByType(self):
        
        out = pd.DataFrame()#columns = self.investments.columns)
        for c in self.investments.columns:
            res = pd.DataFrame()#columns = self.investments.columns)
            for i in self.investments[c].index:
                item_type = self.type_identifier[i[:-2]]
                if (not item_type in res.index) or (not c in res.columns):
                    res.loc[item_type,c] = self.investments.loc[i,c]
                else: 
                    res.loc[item_type,c] += self.investments.loc[i,c]
            out = pd.concat([out,res], axis = 1)
        return out
    
    def plotInvByType(self, plotType = 'pie'):
        
        
        inv_by_type = self.invByType()
        
        inv_by_type.loc[inv_by_type.index.drop('H2_Storage')].plot(
                kind = plotType, subplots = True)
        
    def plotAttr(self, res_type, res_attr):
        
        res = getattr(self,res_type)
        
        for i in res.columns.levels[0]:
            if res_attr in res[i].columns:
                res[i][res_attr].plot()
                
                
    def plotMap(self)      :     

        ax = plt.subplot()
        tx = gpd.read_file('..\\geo\\Texas_State_Boundary_Detailed\\Texas_State_Boundary_Detailed.shp')
        m = tx.plot(ax = ax, color='white', edgecolor='black')
        
        buses = pd.read_excel('..\\grid\\13_Bus_Case.xlsx', sheet_name = 'Bus', index_col = 0)
        buses['Coordinates'] = list(zip(buses['Lon'], buses['Lat']))
        
        buses['Coordinates'] = buses['Coordinates'].apply(Point)
        
        buses_gdf = gpd.GeoDataFrame(buses, geometry = 'Coordinates')
        buses_gdf.plot(ax = ax, color = 'r')
        
        line_gdf = gpd.GeoDataFrame(self.data.line)
        lines = []
        for l in range(len(line_gdf.index)):
            from_point = buses_gdf.loc[line_gdf.iloc[l].From].Coordinates
            to_point = buses_gdf.loc[line_gdf.iloc[l].To].Coordinates
            line = LineString(from_point.coords[:]+to_point.coords[:])
            lines.append(line)
        line_gdf['geometry'] = lines
        line_gdf.plot(ax = ax, color = 'k')
        