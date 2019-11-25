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
import seaborn as sns

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
        
        self.model = dim.deterministicModel(time_data, dirs,
                                            mutables = {self.param: True})
        
    def runMetaModel(self):
        
        self.model.buildModel()
        
        for i in self.range:
            
            setattr(self.model.detModelInstance, self.param, i)
            print('Solving for ', self.param, ' = ', getattr(self.model.detModelInstance, self.param).value)
            
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
        
    def plotStorageEnergy(self):
        
        r = pd.DataFrame()
        for n, i in enumerate(self.res):
            param_val = np.round(self.range[n],4)
            r[param_val] = i.invByType().T.sum().loc[['H2_Storage', 'Battery Energy']]
        
        
    def plotEnergySumByType(self, plotType = 'bar'):
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
        
    def getStorageDuration(self):
        
        r = pd.DataFrame()
        for p,res in zip(self.range,self.res):
            for i in res.plant_inv.index:
                n = i[-2:]
                if i[:-2] == 'ESP':
                    r.loc[int(n),p] = res.plant_inv.loc['ESE'+n].sum()/res.plant_inv.loc['ESP'+n].sum()
        return r
        
    def plotBatteryStorageDuration(self):
        
        plt.figure()
        dur = self.getStorageDuration()
        dur.plot(kind = 'hist', alpha = 0.5, fontsize = 14)
        
        ax = plt.gca()
        ax.set_ylabel('Numbers of Batteries', fontsize = 14)
        ax.set_xlabel('Relative storage [Hours]', fontsize = 14)
        
    def getHydrogenStorageDuration(self):
        
        r = pd.DataFrame()
        for p,res in zip(self.range,self.res):
            for i in res.plant_inv.index:
                n = i[-2:]
                if i[:-2] == 'HS'and ('E'+n) in res.plant_inv.index:
                    r.loc[int(n),p] = res.plant_inv.loc['HS'+n].sum()/res.plant_inv.loc['E'+n].sum()
 
        return r

    def plotHydrogenStorageDurationHist(self):
        
        plt.figure()
        dur = self.getStorageDuration()
        dur.plot(kind = 'hist', alpha = 0.5, fontsize = 14)
        
        ax = plt.gca()
        ax.set_ylabel('Numbers of Storage Tanks', fontsize = 14)
        ax.set_xlabel('Relative storage [Hours]', fontsize = 14)
        
    def plotHydrogenStorageDurationHeat(self):
        
        plt.figure()
        dur = self.getStorageDuration()
        sns.set(font_scale=1.4)
        sns.heatmap(dur,cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    cbar_kws={'label': 'Battery Duration [H]'})
        
        ax = plt.gca()
        ax.set_ylabel('Bus nr.', fontsize = 14)
        ax.set_xlabel('CO2 price [$/kg]', fontsize = 14)
    
    def getBatteryInv(self):
        
        bat = []
        for res in self.res:
            bat.append(res.invByType().loc['Battery','new_cap'])
            
        return bat
  
    def getHydrogenStorage(self):
        
        h2_storage = []
        for res in self.res:
            h2_plant = res.data.hydrogen_plant_char.set_index('Type')
            en_rate = h2_plant.loc['Elec','Energy rate [MWh/kg]']
            h2_storage.append(res.invByType().loc['H2_Storage','new_cap']*en_rate )
            
        return h2_storage
        
#    def plotStorageInv(self):
        
        
        
        
            