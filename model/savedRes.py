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
            
        self.type_identifier = {'HS': 'H2_Storage', 'B': 'Bio', 'S': 'Solar',
                           'N': 'Nuclear', 'E': 'Elec', 'C': 'Coal',
                           'G': 'Gas', 'W': 'Wind'}
        
        self.investments.sort_index(axis = 1, inplace = True)
        
        
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
    
    def invByBus(self):
        indx_1 = [int(i) for i in self.bus.columns.levels[0]]
        indx_2 = self.type_identifier.values()
        out = pd.DataFrame(columns = self.investments.columns,
                           index =  pd.MultiIndex.from_product([indx_1,
                                                                indx_2]))
        for b in self.bus.columns.levels[0]:
            for t in self.type_identifier.keys():
                if t == 'H2_Storage':
                    continue  
                for c in self.investments.columns:
                    indx = '%s%.2d' % (t,int(b))
                    out.loc[(int(b),self.type_identifier[t]), c] = self.investments.loc[indx,c]
                    
        out = out[out.sum(axis = 1) != 0]
        out.drop(labels = 'H2_Storage', level = 1, inplace = True)
        out.sort_index(level = 0, inplace = True)
        return out
    
    def energyByType(self):
        
        res = pd.DataFrame()#columns = self.investments.columns)
        for i in self.plant.columns.levels[0]:
            #res = pd.DataFrame()#columns = self.investments.columns)
            for c in self.plant[i].columns:
                item_type = self.type_identifier[i[:-2]]
                if (not item_type in res.index) or (not c in res.columns):
                    res.loc[item_type,c] = self.plant[i,c].sum()
                else: 
                    res.loc[item_type,c] += self.plant[i,c].sum()
            #out = pd.concat([out,res], axis = 1)
        return res
    
    def plotInvByType(self, plotType = 'pie', subplots = True):
        
        
        inv_by_type = self.invByType()
        
        plt.figure('Investments')
        ax = plt.gca()
        inv_by_type.loc[inv_by_type.index.drop('H2_Storage')].plot(
                kind = plotType, subplots = subplots, ax = ax)
        
    def plotInvByBus(self):
        
        df = self.invByBus()
        ncols = len(df.index.levels[0])
        #plotting
        fig, axes = plt.subplots(nrows=1,
                                 ncols= ncols,
                                 sharey=True,
                                 gridspec_kw={'width_ratios': [len(df.loc[i].index) for i in df.index.levels[0]]})
                                 #figsize=(14 / 2.54, 10 / 2.54))  # width, height
        for i, row in enumerate(df.index.levels[0]):
            ax = axes[i]
            df.loc[(row,)].plot(ax=ax, kind='bar', width=.8 , stacked = True)
        
            ax.set_xlabel(row, weight='bold')
            ax.xaxis.set_label_coords(0.5,-0.2)
            #ax.yaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=.4)
            ax.set_axisbelow(True)
            if i  != (len(df.index.levels[0])-1):
                #ax.spines['left'].set_visible(False)
                ax.legend_.remove()
            else:
                ax.legend(['Initial capacity', 'New capacity'])
            #ax.spines['right'].set_visible(False)
            #ax.spines['top'].set_visible(False)
            
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
                
        
        #make the ticklines invisible
        ax.tick_params(axis=u'x', which=u'x', length=0)
        plt.tight_layout()
        # remove spacing in between
        fig.subplots_adjust(wspace=0)  # space between plots
        
        plt.show()
        
    def plotAttr(self, res_type, res_attr):
        
        res = getattr(self,res_type)
        
        for i in res.columns.levels[0]:
            plt.figure(res_type + '_' + i)
            if res_attr in res[i].columns:
                res[i][res_attr].plot()
                
                
    def plotMap(self, plotLineType = 'Both')      :     
        #fig = plt.figure('Map')
        fig, ax = plt.subplots()
        fig.canvas.set_window_title('Map')
        
        tx = gpd.read_file('..\\geo\\Texas_State_Boundary_Detailed\\Texas_State_Boundary_Detailed.shp')
        tx.plot(ax = ax, color='white', edgecolor='black')
        
        buses = pd.read_excel('..\\grid\\13_Bus_Case.xlsx', sheet_name = 'Bus', index_col = 0)
        buses['xy'] = list(zip(buses['Lon'], buses['Lat']))
        
        buses['Coordinates'] = buses['xy'].apply(Point)
        
        buses_gdf = gpd.GeoDataFrame(buses, geometry = 'Coordinates')
        buses_gdf.plot(ax = ax, color = 'r')
        for idx, row in buses_gdf.iterrows():
            ax.annotate(s = idx, xy = row['xy'], color = 'b', fontsize = 14)
        
        line_gdf = gpd.GeoDataFrame(self.data.line)
        lines = []
        for l in range(len(line_gdf.index)):              
            from_point = buses_gdf.loc[line_gdf.iloc[l].From].Coordinates
            to_point = buses_gdf.loc[line_gdf.iloc[l].To].Coordinates
            line = LineString(from_point.coords[:]+to_point.coords[:])
            lines.append(line)
        line_gdf['geometry'] = lines
        if plotLineType == 'Both':
            line_gdf.plot(ax = ax, column='Cap', cmap='hot')
        else:
            line_gdf.loc[line_gdf.Type == plotLineType ].plot(ax = ax,
                         column='Cap', cmap='hot')
        
        ax = plt.gca()
        ax.axis('off')
        sm = plt.cm.ScalarMappable(cmap='hot',
                                   norm=plt.Normalize(vmin=line_gdf.Cap.min(),
                                                      vmax=line_gdf.Cap.max()))
        sm._A = []
        fig.colorbar(sm)
        plt.tight_layout()
        plt.show()
        
        
    def plotHydrogenSource(self, plotType = 'pie'):
        
        nplots = len(self.hydrogen.columns.levels[0])
        fig, axes = plt.subplots(2,int(np.ceil(nplots/2)))
        for n, i in enumerate(self.hydrogen.columns.levels[0]):
            if n < int(np.ceil(nplots/2)):
                row = 0
                col = n
            else:
                row = 1
                col = n - int(np.ceil(nplots/2))
            direct = self.hydrogen[i].hydrogen_direct.sum()
            storage = self.hydrogen[i].hydrogen_from_storage.sum()
            ng = self.hydrogen[i].hydrogen_import.sum()
            ax = axes[row,col]
            if plotType == 'pie':
                ax.pie([direct, storage, ng])
            elif plotType == 'bar':
                ax.bar([direct, storage, ng])
            ax.set_xlabel(i)
        fig.legend(['direct','storage','natural gas'])
        
    def plotStorageLevel(self):
        
        for n, i in enumerate(self.hydrogen.columns.levels[0]):
            
            if self.hydrogen[i].storage_level.max() > 1E-3:
                fig = plt.figure(i)
                ax = plt.gca()
                self.hydrogen[i].storage_level.plot(ax = ax)
            
        