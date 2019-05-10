# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:42:36 2018

@author: espenfb
"""

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta 
import data_series_tools as dst
import production_records as pr
import parseFunctions as pf

import matplotlib.pyplot as plt

GW2MW = 1000
KW2MW = 0.001

class systemData(object):
    
    def __init__(self, dirs):
        
        delim = ','
        
        for k in dirs.keys():
            setattr(self, k, dirs[k])
        
        self.ctrl_data = pd.read_csv(self.data_dir + self.ctrl_data_file,
                                     delimiter = delim, skipinitialspace=True)
        ctrl_data = self.ctrl_data
        
        
        for i in self.ctrl_data.index:
            data_type = ctrl_data.loc[i,'type']
            data_file = ctrl_data.loc[i,'file']
            data = pd.read_csv(self.data_dir + data_file,
                               skipinitialspace=True)
            
            setattr(self, data_type, data)
                
                
        
        #self.wind_cap['id'] = [ 'W'+str(i) for i in self.wind_cap.index.tolist()]
            
#            
#        self.bus = pd.read_csv(self.data_dir + ctrl_data.loc[ctrl_data.type == 'bus','path'].values[0],
#                                    skipinitialspace=True)
#        self.bus['id'] = ['S'+str(i) for i in self.bus.index.tolist()]
#        
#        self.line = pd.read_csv(self.data_dir + ctrl_data.loc[ctrl_data.type == 'line','path'].values[0],
#                                     skipinitialspace=True)
#        
#        self.wind_cap = pd.read_csv(self.data_dir + 
#                                 ctrl_data.loc[ctrl_data.type =='wind_cap','path'].values[0],
#                                 skipinitialspace=True, index_col = 0)
#        self.wind_cap['id'] = [ 'W'+str(i) for i in self.wind_cap.index.tolist()]
#        
#        
#        self.solar_cap = pd.read_csv(self.data_dir + ctrl_data.loc[ctrl_data.type == 'solar_cap','path'].values[0],
#                                     skipinitialspace=True, index_col = 0)
#        self.solar_cap['id'] = ['S'+str(i) for i in self.solar_cap.index.tolist()]
#
#        self.hydrogen_load = pd.read_csv(self.data_dir + ctrl_data.loc[ctrl_data.type == 'hydrogen_load','path'].values[0],
#                                         skipinitialspace=True, index_col = 0)
#        self.hydrogen_load['id'] = ['C'+str(i) for i in self.hydrogen_load.index.tolist()]
#
#        self.installed = pd.read_csv(self.data_dir + ctrl_data.loc[ctrl_data.type == 'installed','path'].values[0],
#                                         skipinitialspace=True)
#        self.installed['id'] = ['E'+str(i) for i in self.hydrogen.bus_indx]

        self.series = pd.read_csv(self.data_dir + ctrl_data.loc[ctrl_data.type == 'series','file'].values[0],
                                       skipinitialspace=True, index_col = 0)
       
        for i in self.series.index:
            s_type = self.series.loc[i,'type']
            s_path = self.series.loc[i,'path']
            data = pd.read_csv(self.data_dir + s_path, index_col = 0,
                                     delimiter = delim, skipinitialspace=True)
            setattr(self, s_type +'_series', data)
            
        
        
        
    def plotSeries(self, series_type, x_range = 'all', save = False, import_dir = None):
        
        res = getattr(self, series_type)
        
        for k in res.columns:
            plt.figure(str(k))
            ax = plt.gca()
            if x_range == 'all':
                line, = ax.plot(res.loc[:,k])
                line.set_label(k + '_' + series_type)
            else:
                line, = ax.plot(res.loc[pd.date_range(start = x_range[0].tz_convert(None),
                                                      end = x_range[-1].tz_convert(None),
                                                      freq = 'H'),k].tolist())
                line.set_label(k + '_' + series_type)
            plt.legend()
            
            if save:
                plt.savefig(import_dir + str(k) +'.pdf')
    
