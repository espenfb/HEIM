# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:31:44 2019

@author: espenfb
"""


import detInvModel as dim
import savedRes as sr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class metaModel(object):
    
    
    def __init__(self, time_data, dirs, meta_data):
        
        # Times
        for k in time_data.keys():
            setattr(self, k, time_data[k])
        
        # Directories
        for k in dirs.keys():
            setattr(self, k, dirs[k])
            
        # Meta data
        for k in meta_data.keys():
            setattr(self, k, meta_data[k])
        
        self.model = dim.deterministicModel(time_data, dirs)
        
        
        
    def runMetaModel(self, 
                     orientation = 'col', key_col = None, col_value = None):
        
        if orientation == 'col':
            self.type = getattr(self.model.data, self.type)
        elif orientation == 'row':
            self.type = getattr(self.model.data, self.type)
            key_row = (getattr(self.type, key_col) == self.param)
            indx = self.param_type.index[key_row]
        
        for i in self.range:
            
            if orientation == 'col':
                setattr(self.param_type, self.param, i)
            elif orientation == 'row':
                self.param_type.loc[indx,col_value] = i
            
            self.model.buildModel()
            
            self.model.solve()
    
            self.model.processResults()
    
            self.model.saveRes(self.res_dir + 'Result' + '_' + self.param + '_' + str(i) + '\\')
            
    def loadRes(self):
        
        self.res = []
        
        for i in self.range:
            
            self.res.append(sr.savedRes(self.res_dir + 'Result' + '_' +
                                        self.param + '_' + str(i) + '\\',
                                        data = self.model.data))



    def plotH2Sorce(self, plotType = 'bar'):
        
        r = pd.DataFrame()
        for n, i in enumerate(self.res):
            param_val = np.round(self.range[n],4)
            r[param_val] = i.getH2SourceBus().sum()
        
        r[r < 0] = 0
        r.T.plot(kind = plotType)
        
        
    def plotInvByType(self, plotType = 'bar'):
        r = pd.DataFrame()
        for n, i in enumerate(self.res):
            param_val = np.round(self.range[n],4)
            r[param_val] = i.invByType().T.sum().drop(['H2_Storage', 'Battery Energy'])
            
        r.T.plot(kind = plotType) 
        
    def plotenergySumByType(self, plotType = 'bar'):
        r = pd.DataFrame()
        for n, i in enumerate(self.res):
            param_val = np.round(self.range[n],4)
            r[param_val] = i.energySumByType()['prod']
            
        r.T.plot(kind = plotType)
        
    def getTotalEmissions(self):
        out = pd.DataFrame()
        for n, res in enumerate(self.res):
            param = self.range[n]
            out.loc[param, 'Emissions [ton CO2]'] = res.emissionByType().sum() +\
            res.emissionFromH2()
        return out
    
    def plotTotalEmissions(self, figure_name = None):
        ''' Plots total CO2 emissions for the system in tons. '''
        
        emissions = self.getTotalEmissions()
        #if figure_name != None:
        plt.figure(figure_name)
        ax = plt.gca()
        (emissions/1E3).plot(ax = ax)
#        else:
#            (out/1E3).plot()
        
        
    def getPriceStats(self):
        
        out_mean = pd.DataFrame()
        out_std = pd.DataFrame()
        for n, res in enumerate(self.res):
            param = self.range[n]
            for b in res.bus.columns.levels[0]:
                out_mean.loc[int(b),param] = res.bus[b]['nodal_price'].mean()
                out_std.loc[int(b),param] = res.bus[b]['nodal_price'].std()
                
        return out_mean, out_std
    
    def plotPrice(self):
        
        price_mean, price_std = self.getPriceStats()
        
        plt.figure('Mean price')
        ax = plt.gca()
        price_mean.sort_index().T.plot(ax = ax)
        
        plt.figure('Std price')
        ax = plt.gca()
        price_std.sort_index().T.plot(ax = ax)
        