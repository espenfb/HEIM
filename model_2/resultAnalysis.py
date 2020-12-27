# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:14:45 2020

@author: espenfb
"""

import metaRes as mr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
idx = pd.IndexSlice
import matplotlib.pylab as pylab
fs = 15
params = {'legend.fontsize': fs,
          'legend.title_fontsize':fs,
          #'figure.figsize': (15, 4.5),
         'axes.labelsize': fs,
         'axes.titlesize':fs,
         'xtick.labelsize':fs,
         'ytick.labelsize':fs,
         'hatch.linewidth': 1.0 }
pylab.rcParams.update(params)

class resultAnalysis(object):

    def __init__(self, res_dirs, meta_data, sep = '\\', data = None):
        
        for k in meta_data.keys():
            setattr(self, k, meta_data[k])
        
        self.analysis = {}
        for k,v in res_dirs.items():
            meta_res = mr.metaRes(v, meta_data, sep = sep, data = data)
            self.analysis[k] = meta_res
            
            
    def getValue(self, val, relative = False, aggr = False, unstack = True,
                       axis = 1, lower_lim = 0, **kwargs):
        
        out = pd.DataFrame()
        for k,v in self.analysis.items():
            

            if aggr == 'sum':
                vbt = v.getValue(val, **kwargs).sum()
            elif aggr == 'mean':
                vbt = v.getValue(val, **kwargs).mean()
            else:
                vbt= v.getValue(val, **kwargs)
                
            if aggr:
                vbt.name = k
                vbt = vbt.to_frame()
            else:
                vbt = pd.concat([vbt], keys = [k], axis = 1)
                
            out = pd.concat([out,vbt], axis = 1)
        
        if relative:
            out = out.divide(out.sum(level = 0), axis= 0, level = 0)
        if unstack and aggr:
            out = out.unstack(level = 1)
            
        return out  
            
    def getValueByType(self, val, relative = True, aggr = False, unstack = True,
                       axis = 1, lower_lim = 0, **kwargs):
        
        out = pd.DataFrame()
        for k,v in self.analysis.items():

            if aggr == 'sum':
                vbt = v.getValueByType(val, **kwargs).sum()
            elif aggr == 'mean':
                vbt = v.getValueByType(val, **kwargs).mean()
            else:
                vbt = v.getValueByType(val, **kwargs)
                
            if aggr:
                vbt.name = k
                vbt = vbt.to_frame()
            else:
                vbt.columns = pd.MultiIndex.from_product([[k], vbt.columns])
            
            out = pd.concat([out,vbt], axis = 1)
        
        if relative:
            out = out.divide(out.sum(level = 0), axis= 0, level = 0)
        if unstack:
            out = out.unstack()
            
        return out        

        
        
    def getProdByType(self, val, coeff, relative = True, aggr = False, unstack = True,
                       axis = 1, lower_lim = 0, **kwargs):
        
        out = pd.DataFrame()
        for k,v in self.analysis.items():

            if aggr == 'sum':
                vbt = v.getProdByType(val, coeff, **kwargs).sum()
            elif aggr == 'mean':
                vbt = v.getProdByType(val, coeff, **kwargs).mean()  
            else:
                vbt = v.getProdByType(val, coeff, **kwargs)
            
            if aggr:
                vbt.name = k
                vbt = vbt.to_frame()
            else:
                vbt.columns = pd.MultiIndex.from_product([[k], vbt.columns])
                
            out = pd.concat([out,vbt], axis = 1)
        
        if relative:
            out = out.divide(out.sum(level = 0), axis= 0, level = 0)
        if unstack:
            out = out.unstack()
            
        return out 
    
    def getWeightedValue(self, val, weight, 
                         val_objects = None, weight_objects = None,
                         val_coeff = None, weight_coeff = None,
                         relative = False, aggr = False, unstack = False,
                         axis = 1, lower_lim = 0, **kwargs):
        
        out = pd.DataFrame()
        for k,v in self.analysis.items():
            
            
            vbt = v.getWeightedValue(val, weight,
                        val_objects = val_objects, weight_objects = weight_objects,
                        val_coeff = val_coeff, weight_coeff = weight_coeff)
            if aggr == 'sum':
                vbt = vbt.sum(axis = 1)
            elif aggr == 'mean':
                vbt = vbt.mean(axis = 1)
            if aggr:
                vbt.name = k
                vbt = vbt.to_frame()
            else:
                vbt.columns = pd.MultiIndex.from_product([[k], vbt.columns])
                
            out = pd.concat([out,vbt], axis = 1)
        
        if relative:
            out = out.divide(out.sum(level = 0), axis= 0, level = 0)
        if unstack:
            out = out.unstack()
            
        return out 
    
    def getInv(self, objects = [], relative = False, aggr = False, unstack = True,
                       axis = 1, lower_lim = 0, **kwargs):
        
        out = pd.DataFrame()
        for k,v in self.analysis.items():
            meta_res = pd.DataFrame()
            for o in objects: 

                vbt = v.getInvByType(objects = o).sum(axis = 1, level = 0).T
                meta_res = pd.concat([meta_res, vbt], axis = 1)
            meta_res.columns = pd.MultiIndex.from_product([[k], meta_res.columns])
            out = pd.concat([out,meta_res], axis = 1)
        
        if relative:
            out = out.divide(out.sum(level = 0), axis = 0, level = 0)
        if unstack and aggr:
            out = out.unstack(level = 1)
            
        return out
    
    def getInvByType(self, cap_type = 'power', objects = None, lower_lim = 0.0,
                     conH2power = False, relative = True, unstack = True):
        
        out = pd.DataFrame()
        for k,v in self.analysis.items():
            vbt = v.getInvByType(cap_type = cap_type, objects = objects , lower_lim = lower_lim,
                     conH2power = conH2power).sum(axis = 1, level = 0)
            
            vbt.columns = pd.MultiIndex.from_product([[k], vbt.columns])
                
            out = pd.concat([out,vbt], axis = 1)
        
        if relative:
            out = out.divide(out.sum(level = 0), axis= 0, level = 0)
        if unstack:
            out = out.unstack()
            
        return out 
        
    
    def getSummary(self):
        
        out = pd.DataFrame()
        for k,v in self.analysis.items():
            s = v.getSummary()
            s.columns = pd.MultiIndex.from_product([[k], s.columns])
            out = pd.concat([out,s], axis = 1)
        return out

    def getStorageDuration(self, objects = 'BATTERY_STORAGE'):
        
        out = pd.DataFrame()
        for k,v in self.analysis.items():
            dur = v.getStorageDuration(objects = objects).mean()
            dur.name = k
            dur = dur.to_frame()
            out = pd.concat([out,dur], axis = 1)
        return out
        
            
    def plotValue(self, val, objects = None, aggr = 'sum', by_node = True,
                        relative = False, unstack = True, plot_type = 'multi',
                        xfilter = None, ax2 = None, no_leg = False, **kwargs):
        
        vbt = self.getValue(val,objects = objects, aggr = aggr, by_node = by_node,
                                  relative = relative, unstack = unstack)
        
        if xfilter: vbt = vbt.iloc[xfilter]
        
        if plot_type == 'multi':
            self.multiPlot(vbt, **kwargs)
        elif plot_type == 'single':
            self.singlePlot(vbt, ax2 = ax2, no_leg = no_leg, **kwargs)
            
    def plotValueByType(self, val, objects = None, aggr = 'sum',
                        relative = True, unstack = True, plot_type = 'multi',
                        color_dict = False, **kwargs):
        
        vbt = self.getValueByType(val,objects = objects, aggr = aggr,
                                  relative = relative, unstack = unstack)
        if plot_type == 'multi':
            self.multiPlot(vbt, color_dict = color_dict, **kwargs)
        elif plot_type == 'single':
            self.singlePlot(vbt, **kwargs)
        
    def plotProdByType(self, val, coeff, objects = None, aggr = 'sum',
                       relative = True, unstack = True, plot_type = 'multi',
                       leg_at_fig = True, color_dict = False, **kwargs):
        
        vbt = self.getProdByType(val,coeff, objects = objects, aggr = aggr,
                                 relative = relative, unstack = unstack)
        
        if plot_type == 'multi':
            self.multiPlot(vbt, color_dict = color_dict, **kwargs)
        elif plot_type == 'single':
            self.singlePlot(vbt, **kwargs)
            
            
    def plotWeightedValue(self, val, weight,
                          val_objects = None, weight_objects = None,
                          weight_coeff = None, relative = False,
                          unstack = False,
                          aggr = 'mean', plot_type = 'single', xfilter = None, **kwargs):
        
        weighted_mean = self.getWeightedValue(val,weight,
                                          val_objects = val_objects,
                                          weight_objects = weight_objects,
                                          weight_coeff = weight_coeff,
                                          relative = relative, unstack = unstack)
        
        mean = self.getValue(val, objects = val_objects,
                             by_node = True, aggr = aggr)
        
        values = (weighted_mean/mean).mean(axis = 1, level = 0)
        
        if xfilter: values = values.iloc[xfilter]
         
        if plot_type == 'multi':
            self.multiPlot(values, **kwargs)
        elif plot_type == 'single':
            self.singlePlot(values, **kwargs)
   
    def plotInv(self, objects = [], plot_type = 'single', ax2 = None, **kwargs):
        
        vbt = self.getInv(objects = objects).sum(axis = 1, level = 0)
        
        if plot_type == 'multi':
            self.multiPlot(vbt, **kwargs)
        elif plot_type == 'single':
            self.singlePlot(vbt, ax2 = ax2, **kwargs)            
            
    def plotSummary(self, by_col = False, xlabel = '', **kwargs):
        
        s = self.getSummary()
        
        for i in s.columns.get_level_values(1).unique():
            if i == by_col: continue
            df = s.loc[idx[:],idx[:,[by_col,i]]]
            if by_col:
                out = pd.DataFrame()
                for j in s.columns.get_level_values(0).unique():
                    t = df[j]
                    t= t.set_index(by_col)
                    t.columns = pd.MultiIndex.from_product([[j],t.columns])
                    out = pd.concat([out,t])
                df = out
            df = df.droplevel(axis = 1, level = 1)
            if by_col: xlabel =  by_col
            self.singlePlot(df, **kwargs, ylabel = i, xlabel = xlabel)
        
    def plotSummaryType(self,val, by_col = False, xlabel = '',
                        ax2 = None, no_leg = False, **kwargs):
        
        s = self.getSummary()
        vbt = s.loc[idx[:],idx[:,val]].droplevel(axis = 1, level = 1)
        
        self.singlePlot(vbt, ax2 = ax2, ylabel = val, xlabel = xlabel,
                        no_leg = no_leg, **kwargs)
            
    def plotStorageDuration(self, objects = 'BATTERY_STORAGE',
                            plot_type = 'single',  **kwargs):
        
        vbt = self.getStorageDuration(objects = objects)
        
        if plot_type == 'multi':
            self.multiPlot(vbt, **kwargs)
        elif plot_type == 'single':
            self.singlePlot(vbt, **kwargs)
        
    def multiPlot(self, vbt, kind = 'bar', leg_col = 3,
                xscale = False, yscale = False,
                sharex=True, sharey=True, bar_labels = False, bbox = None,
                xlabel = '', ylabel = '', lower_lim = False,
                first_base = False, leg_loc = 'upper center', color_dict = False,
                **kwargs):
        dim = (1,vbt.columns.levels[0].shape[0])
        if sharey: vbt_max = vbt.sum(axis = 1, level = 0).max().max()
        f,a = plt.subplots(dim[0],dim[1], sharex=sharex, sharey=sharey,
                           figsize=(5*dim[1], 5*dim[0]))
        for n, c in enumerate(vbt.columns.levels[0]):
            vbt_c = vbt[c]
            if yscale: vbt_c *= yscale
            if xscale: vbt_c.index = [int(i*xscale) for i in vbt_c.index]
            if lower_lim: vbt_c = vbt_c.loc[:,vbt_c.max() > lower_lim]
            if color_dict: 
                order = list(color_dict.keys())
                order = [i for i in order if i in vbt_c.columns]
                if any(vbt_c.columns.isin(order) == False): order = order +  vbt_c.columns[vbt_c.columns.isin(order) == False].to_list()
                vbt_c = vbt_c[order]
                vbt_c.plot(kind = kind, title = c, ax = a[n],
                           color=[color_dict.get(x, '#333333') for x in vbt_c.columns],
                                  **kwargs)
            else:
                vbt_c.plot(kind = kind, title = c, ax = a[n], **kwargs)
            a[n].set_ylim([0,vbt_c.sum(axis = 1).max()*1.05])
            if bar_labels:
                vbt_sum = vbt_c.sum(axis = 1)
                if not sharey: vbt_max = vbt_sum.max()
                base = vbt_sum.iloc[0] if not first_base else vbt.sum(axis = 1, level = 0).iloc[0,0]*yscale
                for e, (k, v) in enumerate(vbt_sum.items()):
                    a[n].annotate('%d%%' % np.round(v/base*100),
                                  (e-0.35, v + 0.01*vbt_max ),
                                  fontsize = 13)
    
            a[n].spines['right'].set_visible(False)
            a[n].spines['top'].set_visible(False)
            a[n].get_legend().remove()
        labels = a[0].get_legend_handles_labels()[1]
        f.legend(labels,loc = leg_loc, ncol = leg_col, frameon=False, bbox_to_anchor = bbox)
        plt.setp(a[:], xlabel=xlabel)
        plt.setp(a[0], ylabel=ylabel)
        plt.tight_layout()
                
    def singlePlot(self, vbt, kind = 'bar',
                xscale = False, yscale = False,  sum_types = False,
                sharex=True, sharey=True, bar_labels = False,
                xlabel = '', ylabel = '', leg_at_fig = True,
                lower_lim = False, leg_loc = 'best', leg_col = 1,
                plot_grid = True, color_dict = None, ax2 = None,
                set_style = True, no_leg = False, bbox = None, text = None, **kwargs):
        
        if sharey: vbt_max = vbt.sum(axis = 1, level = 0).max().max()
        
        if ax2:
            a = ax2
        else:
            f,a = plt.subplots(1,1, sharex=sharex, sharey=sharey,
                                       figsize=(7, 5))
            leg_at_fig = f

        if yscale: vbt *= yscale
        if xscale: vbt.index = [int(i*xscale) for i in vbt.index]
        if lower_lim: vbt = vbt.loc[:,vbt.max() > lower_lim]
        if sum_types == 'sum':
            vbt = vbt.sum(axis = 1, level = 0)
        elif sum_types == 'mean':
            vbt = vbt.mean(axis = 1, level = 0)
        
        
        if kind == 'scatter':
            for i in vbt.columns:
                plt.scatter(x = vbt.index.to_list(),
                                 y = vbt[i].to_list(),
                                 ax = a, **kwargs)
        elif color_dict: 
                order = list(color_dict.keys())
                order = [i for i in order if i in vbt.columns]
                if any(vbt.columns.isin(order) == False): order = order +  vbt.columns[vbt.columns.isin(order) == False].to_list()
                vbt = vbt[order]
                for x in vbt.columns:
                    a.plot(vbt[x], label = x,
                           color = color_dict.get(x, '#333333'),
                           **kwargs)
        else:
            a.plot(vbt, **kwargs)
            
        if set_style:
    
            if bar_labels:
                vbt_sum = vbt.sum(axis = 1)
                if not sharey: vbt_max = vbt_sum.max()
                base = vbt_sum.iloc[0]
                for e, (k, v) in enumerate(vbt_sum.items()):
                    a.annotate('%d %%' % np.round(v/base*100), (e-0.35, v + 0.01*vbt_max ))
            if text:
                print(text)
                vbt_max = vbt.max().max()
                indx_min = vbt.index.min()
                props = dict(boxstyle='round', facecolor='white', edgecolor = 'grey', alpha=1.0)
                a.text(0.05,0.9, text,
                       transform= a.transAxes, bbox = props)
        
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
            if not no_leg:
                if leg_at_fig:
                    leg = a.get_legend()
                    if leg: leg.remove()
                    leg = leg_at_fig.legend(loc = leg_loc, ncol = leg_col, bbox_to_anchor = bbox)
                    
                else:
                    leg = a.legend(loc = leg_loc, frameon = False, ncol = leg_col, bbox_to_anchor = bbox)
                #leg.get_frame().set_linewidth(0.0)
            if plot_grid: a.grid(linestyle='--', linewidth='1.0', color='lightgrey', which = 'both')
            plt.setp(a, xlabel=xlabel)
            plt.setp(a, ylabel=ylabel)
            #plt.tight_layout()
        
        
    def plotPriceIQR(self,  ax2 = None, no_leg = False, **kwargs ):
        
        
        price = self.getValue('energyBalance', objects = 'EL_NODES', by_node=True)
        price_q = price.quantile([0.25,0.75]).mean(axis = 1, level = [0,1])
        iqr = (price_q.iloc[1]-price_q.iloc[0]).unstack().T
        
        self.singlePlot(iqr, leg_at_fig = False, no_leg = no_leg, ax2 = ax2, **kwargs)       