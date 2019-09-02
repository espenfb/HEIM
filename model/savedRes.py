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
import copy

from ast import literal_eval
        
        
class savedRes(object):
    
    def __init__(self, folder, data = None):
    
        for file in os.listdir(folder):
            if file[:-4] == 'plant_inv' or file[:-4] == 'line_inv':
                df = pd.read_csv(folder + file, index_col = 0, header = [0]).replace(np.NaN,0)
            else:
                df = pd.read_csv(folder + file, index_col = 0, header = [0,1]).replace(np.NaN,0)
                    
            setattr(self, file[:-4], df)
            
        self.data = data
            
        self.type_identifier = {'HS': 'H2_Storage', 'B': 'Bio', 'S': 'Solar',
                           'N': 'Nuclear', 'E': 'Elec', 'C': 'Coal',
                           'G': 'Gas', 'W': 'Wind'}
        
        self.plant_inv.sort_index(axis = 1, inplace = True)
        self.addGeoBus()
        self.addGeoLine()
        
        
    def invByType(self):
        
        out = pd.DataFrame()
        for c in self.plant_inv.columns:
            res = pd.DataFrame()
            for i in self.plant_inv[c].index:
                item_type = self.type_identifier[i[:-2]]
                if (not item_type in res.index) or (not c in res.columns):
                    res.loc[item_type,c] = self.plant_inv.loc[i,c]
                else: 
                    res.loc[item_type,c] += self.plant_inv.loc[i,c]
            out = pd.concat([out,res], axis = 1)
        return out
    
    def invByBus(self):
        indx_1 = [int(i) for i in self.bus.columns.levels[0]]
        indx_2 = self.type_identifier.values()
        out = pd.DataFrame(columns = self.plant_inv.columns,
                           index =  pd.MultiIndex.from_product([indx_1,
                                                                indx_2]))
        for b in self.bus.columns.levels[0]:
            for t in self.type_identifier.keys():
                if t == 'H2_Storage':
                    continue  
                for c in self.plant_inv.columns:
                    indx = '%s%.2d' % (t,int(b))
                    out.loc[(int(b),self.type_identifier[t]), c] = self.plant_inv.loc[indx,c]
                    
        out = out[out.sum(axis = 1) != 0]
        out.drop(labels = 'H2_Storage', level = 1, inplace = True)
        out.sort_index(level = 0, inplace = True)
        return out
    
    def energyByType(self):
        
        res = pd.DataFrame()
        for i in self.plant.columns.levels[0]:
            for c in self.plant[i].columns:
                item_type = self.type_identifier[i[:-2]]
                if (not item_type in res.index) or (not c in res.columns):
                    res.loc[item_type,c] = self.plant[i,c].sum()
                else:
                    if np.isnan(res.loc[item_type,c]):
                        res.loc[item_type,c] = self.plant[i,c].sum()
                    else:
                        res.loc[item_type,c] += self.plant[i,c].sum()
        return res
    
    def plotEnergyByType(self):
        energyByType = self.energyByType()
        energyByType.plot(kind = 'bar')
    
    def plotInvByType(self, plotType = 'bar', subplots = False):
        
        
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
        
        
    def getH2SourceBus(self):
        
        df = pd.DataFrame(columns = ['Direct','Storage','Natural Gas'])
        for i, row in enumerate(self.hydrogen.columns.levels[0]):
            df.loc[row,'Direct'] = self.hydrogen[row].hydrogen_direct.sum()
            df.loc[row,'Storage'] = self.hydrogen[row].hydrogen_from_storage.sum()
            df.loc[row,'Natural Gas'] = self.hydrogen[row].hydrogen_import.sum()
        return df
             
    def plotH2ByBus(self, plotType = 'bar'):
        
        df = self.getH2SourceBus()
        
        df.plot(kind = plotType, stacked = True)
        plt.legend(['Direct','Storage','Natural Gas'])
        
    def plotAttr(self, res_type, res_attr):
        
        res = getattr(self,res_type)
        
        for i in res.columns.levels[0]:
            plt.figure(res_type + '_' + i)
            if res_attr in res[i].columns:
                res[i][res_attr].plot()
                


    def getLineRes(self):
        self.addGeoLine(LineType = 'res')  
        line_res_geo = gpd.GeoDataFrame(self.line_inv.groupby(['From','To']).agg({'Cap':'sum', 'geometry': 'first'})  )
        return line_res_geo    
    
    def addGeoLine(self, LineType = 'data'):
        
        if LineType == 'data':
            self.data.line.sort_index(inplace = True)
            line_data = self.data.line
        elif LineType == 'res':
            self.line_inv.sort_index(inplace = True)
            line_data = self.line_inv
        
        if 'geometry' not in line_data.columns:
            line_gdf = gpd.GeoDataFrame(line_data)
            lines = []
            for l in range(len(line_gdf.index)):              
                from_point = self.data.bus.loc[line_gdf.iloc[l].From].Coordinates
                to_point = self.data.bus.loc[line_gdf.iloc[l].To].Coordinates
                line = LineString(from_point.coords[:]+to_point.coords[:])
                lines.append(line)
            line_gdf['geometry'] = lines
            
        if LineType == 'data':
            self.data.line = line_data
        elif LineType == 'res':
             self.line_inv = line_data
        
    def addGeoBus(self):
        if 'Coordinates' not in self.data.bus.columns:
            buses = self.data.bus.set_index('Bus')
            buses['xy'] = list(zip(buses['Lon'], buses['Lat']))
            
            buses['Coordinates'] = buses['xy'].apply(Point)
            
            self.data.bus = gpd.GeoDataFrame(buses, geometry = 'Coordinates')

    def plotMap(self, plotLineType = ['Both'], node_color = 'k', colormap = 'tab20c')      :     
        #fig = plt.figure('Map')
        fig, axes = plt.subplots(nrows=1,
                                 ncols= len(plotLineType),
                                 sharey=True, constrained_layout=True)
        fig.canvas.set_window_title('Map')# + ' - ' + plotLineType)
        
        for n, i in enumerate(plotLineType):
            if len(plotLineType) > 1:
                ax = axes[n]
            else:
                ax = axes
            ax.set_title(i)
            tx = gpd.read_file('..\\geo\\Texas_State_Boundary_Detailed\\Texas_State_Boundary_Detailed.shp')
            tx.plot(ax = ax, color='white', edgecolor='black')
            
            line_data_limits = self.data.line.groupby(['From','To','Type']).agg({'Cap':'sum'})
            vmin = line_data_limits.Cap.min()
            vmax = line_data_limits.Cap.max()
            
            self.data.bus.plot(ax = ax, color = node_color)
            for idx, row in self.data.bus.iterrows():
                ax.annotate(s = idx, xy = row['xy'], color = 'b', fontsize = 14)
                
            if i == 'Both':
                line_data = self.data.line.groupby(['From','To']).agg({'Cap':'sum', 'geometry': 'first'})
                line_data = gpd.GeoDataFrame(line_data)
                line_data.plot(ax = ax, column='Cap', cmap=colormap, vmin=vmin, vmax=vmax)
            elif i == 'Res':
                line_data = self.getLineRes()
                line_data.plot(ax = ax, column='Cap', cmap=colormap, vmin=vmin, vmax=vmax)
            elif i == 'Total':
                line_data = self.getLineRes()
                b = self.data.line.groupby(['From','To','Type']).agg({'Cap':'sum', 'geometry': 'first'}).reset_index('Type')
                line_data = pd.concat([line_data, b[b.Type == 'Existing']], sort = True)
                line_data = line_data.groupby(['From','To']).agg({'Cap':'sum', 'geometry': 'first'})
                line_data = gpd.GeoDataFrame(line_data)
                line_data.plot(ax = ax, column='Cap', cmap=colormap, vmin=vmin, vmax=vmax)
            else:
                line_data = self.data.line.groupby(['From','To','Type']).agg({'Cap':'sum', 'geometry': 'first'})
                line_data.reset_index(level = 'Type', inplace = True)
                line_data = gpd.GeoDataFrame(line_data)
                line_data.loc[line_data.Type == i ].plot(ax = ax,
                             column='Cap', cmap=colormap, vmin=vmin, vmax=vmax)
        
            #ax = plt.gca()
            ax.axis('off')
            
        plt.tight_layout()
            
        sm = plt.cm.ScalarMappable(cmap=colormap,
                                   norm=plt.Normalize(vmin=vmin,
                                                      vmax=vmax))
        sm._A = []
        
        fig.subplots_adjust(right=0.8)
        cax = fig.add_axes([0.85, 0.1, 0.03, 0.8])
        fig.colorbar(sm, cax = cax)
       # plt.tight_layout()
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
                ax.bar(['direct','storage','natural gas'],[direct, storage, ng])
            ax.set_xlabel(i)
        fig.legend(['direct','storage','natural gas'])
        
    def plotStorageLevel(self):
        
        for n, i in enumerate(self.hydrogen.columns.levels[0]):
            
            if self.hydrogen[i].storage_level.max() > 1E-3:
                plt.figure(i)
                ax = plt.gca()
                self.hydrogen[i].storage_level.plot(ax = ax)
        
    
        
    def getTotalCosts(self):
        
        inv_cost = self.data.inv_cost.set_index('Type').Cost
        new_inv = self.invByType().rename({'Bio':'Biomass'}).new_cap
        
        df = pd.DataFrame(inv_cost*new_inv, columns = ['Generation'])
        
        print(df)

        line_inv_cap = copy.copy(self.line_inv.Cap)
        line_inv_cap.index = [literal_eval(i) for i in line_inv_cap.index]
        line_data = copy.copy(self.data.line)
        line_data.index = zip(line_data.index, line_data.From, line_data.To)
        line_inv_cost = line_data.Cost
        
        line_inv_cost = (line_inv_cap*line_inv_cost).dropna().rename({0:'Line'})
        
        df.loc['Lines','Lines'] = line_inv_cost.sum()
        
#        line_inv_cost = pd.concat([line_inv_cost,self.data.line[['From','To']]], axis = 1)
#        print(line_inv_cost)
        
#        a = pd.DataFrame(line_inv_cost.groupby(['From','To']).agg({0:'sum'}).rename({0:'Line'}))
#        print(a)
#        df = pd.concat([df,a])
        
        return df
        
        
            
        