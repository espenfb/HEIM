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
            
    def getH2Source(self):
        
        r = pd.DataFrame()
        for n, i in enumerate(self.res):
            param_val = np.round(self.range[n],4)
            r[param_val] = i.getH2SourceBus().sum()
        
        r[r < 0] = 0
        return r


    def plotH2Source(self, plotType = 'bar', alpha = 1.0, ax = None, xunit = 'ton'):
        
        if ax == None:
            plt.figure()
            ax = plt.gca()
        
        r = self.getH2Source().T
        if xunit == 'ton':
            r.index = r.index*1000
        
        r.plot(kind = plotType, alpha = alpha, ax = ax)
        
    def getH2Price(self):
        out = pd.DataFrame()
        idx = pd.IndexSlice
        for n, res in enumerate(self.res):
            param_val = np.round(self.range[n],4)
            weights = res.getH2ShareBus()
            prices = res.hydrogen.loc[(idx[:],idx[:,'hydrogen_price'])].mean()
            prices.index = prices.index.droplevel(level = 1)
            out.loc['Hydrogen price',param_val] = (weights*prices).sum()
        return out
        
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
       
    def getEnergySumByType(self):
        r = pd.DataFrame()
        for n, i in enumerate(self.res):
            param_val = np.round(self.range[n],4)
            r[param_val] = i.energySumByType()['prod']
            
        return r
        
    def plotEnergySumByType(self, plotType = 'bar', xunit = 'ton'):
        r = pd.DataFrame()
        for n, i in enumerate(self.res):
            param_val = np.round(self.range[n],4)
            r[param_val] = i.energySumByType()['prod']
        
        e_sum = r.T
        if xunit == 'ton':
            e_sum.index = e_sum.index*1000
        
        e_sum.plot(kind = plotType)
        
    def getHydrogenNgEmissions(self):
        out = pd.DataFrame()
        for n, res in enumerate(self.res):
            param = self.range[n]
            out.loc[param, 'Emissions from H2 [ton CO2]'] = res.emissionFromH2() # kg
        return out    
    
    def getPowerSystemEmissions(self):
        out = pd.DataFrame()
        for n, res in enumerate(self.res):
            param = self.range[n]
            out.loc[param, 'Emissions from PS [CO2]'] = res.emissionByType().sum() # kg
        return out
        
    def getTotalEmissions(self):
        out = pd.DataFrame()
        for n, res in enumerate(self.res):
            param = self.range[n]
            out.loc[param, 'Tot Emissions [ton CO2]'] = res.emissionByType().sum() +\
            res.emissionFromH2() # kg
        return out
    
    def plotTotalEmissions(self, figure_name = None):
        ''' Plots total CO2 emissions for the system in tons. '''
        
        emissions = self.getTotalEmissions()
        #if figure_name != None:
        plt.figure(figure_name)
        ax = plt.gca()
        (emissions/1E3).plot(ax = ax)
        ax.set_ylabel('CO$_2$ emission [Ton]', fontsize = 12)
        ax.set_xlabel('CO$_2$ price [\$/kg]', fontsize = 12)
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
        
        out = pd.DataFrame()
        for n,res in enumerate(self.res):
            param = self.range[n]
            out.loc[param, 'Battery'] = res.invByType().loc['Battery','new_cap']
            
        return out
  
    def getHydrogenStorage(self):
        
        h2_storage = []
        for res in self.res:
            h2_plant = res.data.hydrogen_plant_char.set_index('Type')
            en_rate = h2_plant.loc['Elec','Energy rate [MWh/kg]']
            h2_storage.append(res.invByType().loc['H2_Storage','new_cap']*en_rate )
            
        return h2_storage
    
    def getLineInv(self, new_cap_only = True):
        
        if new_cap_only:
            init = 0
        else:
            line_data = self.res[0].data.line
            init = line_data[line_data.Type == 'Existing'].Cap.sum()
        
        out = pd.DataFrame()
        for n,r in enumerate(self.res):
            param = self.range[n]
            out.loc[param, 'Transmission'] = init + r.line_inv.Cap.sum()
        return out
#    def plotStorageInv(self):
        
        
    def getMetaStat(self):
        
        idx = pd.IndexSlice
        
        out = pd.DataFrame()
        for n,r in enumerate(self.res):
            param = self.range[n]
            prod = r.plant.loc[idx[:],idx[:,'prod']].sum().sum()
            cur = r.plant.loc[idx[:],idx[:,'cur']].sum().sum()
            out.loc[param, 'curtailment [%]'] = (cur/(cur + prod))*100
            out.loc[param, 'price mean [$/MWh]'] = r.bus.loc[idx[:],idx[:,'nodal_price']].mean().mean()
            out.loc[param, 'price std [$/MWh]'] = r.bus.loc[idx[:],idx[:,'nodal_price']].std().mean()        
        return out
    
    def getElecOverSizing(self):
        
        out = pd.DataFrame()
        for n,r in enumerate(self.res):
            param = self.range[n]
            out.loc[:,param] = r.getElecOverSizing() 
        return out
    
    def getH2StorageDur(self):
        
        out = pd.DataFrame()
        for n,r in enumerate(self.res):
            param = self.range[n]
            out.loc[:,param] = r.getH2StorageDur()
        return out
            