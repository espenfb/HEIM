# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:44:50 2020

@author: espenfb
"""

import savedRes as sr

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

idx = pd.IndexSlice

class metaRes(object):
    
    def __init__(self, res_dir, meta_data, sep = '\\', data = None):
        
        self.res_dir = res_dir
        self.data = data

        for k in meta_data.keys():
                setattr(self, k, meta_data[k])
                
        self.loadRes()

    def loadRes(self):
        
        self.res = []
        
        for i in self.range:
            
            self.res.append(sr.savedRes(self.res_dir + 'Result' + '_' +
                                        self.param + '_' + str(i) + '\\',
                                        data = self.data))
            
    def getH2Source(self):
        
        r = pd.DataFrame()
        for n, i in enumerate(self.res):
            param_val = np.round(self.range[n],4)
            r[param_val] = i.getH2SourceBus().sum()
        
        r[r < 0] = 0
        return r
    
    def getValue(self, val, objects = None, times = idx[:],
                       lower_lim = 0, by_node = False):
                
        out = pd.DataFrame()
        for n, res in enumerate(self.res):
            param_val = np.round(self.range[n],4)
            res = res.getValue(val, objects = objects, times = times,
                               lower_lim = lower_lim, by_node = by_node)
            res.columns = pd.MultiIndex.from_product([[param_val],res.columns])
            out = pd.concat([out,res], axis = 1)
        return out
    
    def getValueByType(self, val, objects = None, times = idx[:],
                       lower_lim = 0, sort_by_cf = True):
                
        out = pd.DataFrame()
        for n, res in enumerate(self.res):
            param_val = np.round(self.range[n],4)
            res = res.getValueByType(val, objects = objects,
                                     lower_lim = lower_lim, sort_by_cf = sort_by_cf)
            res.columns = pd.MultiIndex.from_product([[param_val],res.columns])
            out = pd.concat([out,res], axis = 1)
        return out
    
    def getProdByType(self, val_type, coeff, objects = None):
        
        out = pd.DataFrame()
        for n, res in enumerate(self.res):
            param_val = np.round(self.range[n],4)
            res = res.getProdByType(val_type, coeff, objects = objects)
            res.columns = pd.MultiIndex.from_product([[param_val],res.columns])
            out = pd.concat([out,res], axis = 1)
        return out
    
    def getWeightedValue(self, val, weight, val_objects = None, 
                        weight_objects = None , val_coeff = None,
                        weight_coeff = None):
                
        out = pd.DataFrame()
        for n, res in enumerate(self.res):
            param_val = np.round(self.range[n],4)
            res = res.getWeightedValue(val, weight, val_objects = val_objects, 
                                       weight_objects=weight_objects,
                                       val_coeff = val_coeff,
                                       weight_coeff= weight_coeff)
            res = res.to_frame(name = param_val)
            out = pd.concat([out,res], axis = 1)
        return out.T
    
    def plotValueByType(self, val, objects = None, aggr_by = 'sum', lower_lim = 0,
                        kind = 'bar', times = idx[:], scale = 1,
                        xscale = 1, base_val = None, base_obj = idx[:],
                        prcnt = True, color_dict = False, **kwargs):
        
        if aggr_by == 'sum':
            val = self.getValueByType(val, objects = objects).sum()
        elif aggr_by == 'mean':
            val = self.getValueByType(val, objects = objects).mean()
            
        val = val.unstack()
            
        if base_val != None:
            if aggr_by == 'sum':
                base = self.getValueByType(base_val, objects = base_obj, lower_lim = lower_lim).sum().sum(level = 0)
            elif aggr_by == 'mean':
                base = self.getValueByType(base_val, objects = base_obj, lower_lim = lower_lim).mean().mean(level = 0)
            val = val.divide(base, axis = 0)
            if prcnt: val = val*100
        
        val = val*scale
        val.index = val.index*xscale
        
        if color_dict: 
            val.plot(kind = kind,
                     color=[color_dict.get(x, '#333333') for x in val.columns],
                     **kwargs)
        else:
            val.plot(kind = kind, **kwargs) 
        
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
    def getInv(self, cap_type = 'power', objects = None):
        
        out = pd.DataFrame()
        for n, res in enumerate(self.res):
            param_val = np.round(self.range[n],4)
            res = res.getInv(cap_type = cap_type, objects = objects)
            res.columns = pd.MultiIndex.from_product([[param_val],res.columns])
            out = pd.concat([out,res], axis = 1)
        return out

        
    def getInvByType(self, cap_type = 'power', objects = None, lower_lim = 0.0,
                     conH2power = False):
        
        out = pd.DataFrame()
        for n, res in enumerate(self.res):
            param_val = np.round(self.range[n],4)
            res = res.getInvByType(cap_type = cap_type, objects = objects,
                                   lower_lim = lower_lim, conH2power = conH2power)
            res.columns = pd.MultiIndex.from_product([[param_val],res.columns])
            out = pd.concat([out,res], axis = 1)
        return out

    def plotInvByType(self, cap_type = 'power', objects = None, lower_lim = 0,
                      xscale = 1, yscale = 1, conH2power = False, **kwargs):
        
        inv = self.getInvByType(cap_type = cap_type , objects = objects,
                  lower_lim = lower_lim, conH2power = conH2power)
        inv = inv.sum(axis =1,level=0).T*yscale
        inv.index = inv.index*xscale
        inv.plot(**kwargs)

        
    def getStorageDuration(self, objects = 'BATTERY_STORAGE'):
        
        energy = self.getInv(objects = objects, cap_type='energy').sum(axis = 1, level = 0)
        pwr = self.getInv(objects = objects, cap_type='power').sum(axis = 1, level = 0)
        
        return energy/pwr
        
    def getLineInv(self, nodes = 'EL_NODES', cap_type = 'Cap'):
        
        out = pd.DataFrame()
        for n, res in enumerate(self.res):
            param_val = np.round(self.range[n],4)
            res = res.getLineInv(nodes = nodes)[cap_type]
            out = pd.concat([out,res.to_frame(name = param_val)], axis = 1)
        return out
        
        
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
    
    def getElecUtilization(self):
        
        out = pd.DataFrame()
        for n,r in enumerate(self.res):
            param = self.range[n]
            out[param] = r.getElecUtilization()
            
        return out
    
    def getBatteryUtilization(self):
        
        out = pd.DataFrame()
        for n,r in enumerate(self.res):
            param = self.range[n]
            out[param] = r.getBatteryUtilization()
            
        return out
        
    def getH2StorageCap(self):
        
        out = pd.DataFrame()
        for n,r in enumerate(self.res):
            param = self.range[n]
            out[param] = r.getH2StorageCap()
            
        return out
        
    def getElecCap(self):
        
        out = pd.DataFrame()
        for n,r in enumerate(self.res):
            param = self.range[n]
            out[param] = r.getElecCap() # in MW
            
        return out
    
    def getAverageH2PowerPrice(self):
        
        out = pd.DataFrame()
        for n,r in enumerate(self.res):
            param = self.range[n]
            out[param] = r.getAverageH2PowerPrice()
            
        return out
    
    
    def plotMapSelection(self,selection, linetype = None, linestyle = 'color', line_color = 'k',
                         objects = None,
                         nodes = None,
                         cap_type='energy',
                         title_prefix = '', mkr_scaling= 100, marker = 'o',
                         edgecolors = 'k', legend_loc = 'best', ncol = 3, hatch = '',
                         line_lim = 0, bus_lim = 0, labels = False,  alpha = 0.5,
                         fig = False, axes = False, node_color = 'k', plot_shape = True,
                         bbox_node = None, bbox_line = None, fontsize = 22):
        max_node = False
        max_line = False
        if objects:
            inv = self.getInv(cap_type=cap_type,objects= objects)
            max_node = inv.loc[idx[:],idx[selection]].max().max()
        if linetype:
            max_line = self.getLineInv(nodes = nodes).max().max()
        
        if not fig and not axes: fig,axes = plt.subplots(1,len(selection),sharex=True, sharey=True)
        i = 0
        for n,r in enumerate(self.res):
            param = self.range[n]
            if param in selection:
                plt.text(0.76, 0.85, title_prefix + str(int(param*1E3)) + ' \$/tonne CO$_2$',
                         horizontalalignment='center', fontsize = fontsize-1,
                         transform = axes[i].transAxes)
                #axes[i].set_title(title_prefix + str(int(param*1E3)) + ' \$/tonne CO$_2$')
                if param == selection[-1]:
                    r.plotMap(linetype = linetype,  linestyle = linestyle, line_color =line_color,
                              objects = objects, nodes = nodes,
                              line_lim = line_lim, bus_lim = bus_lim, nodetype = cap_type,
                              fig = fig, ax = axes[i], mkr_scaling= mkr_scaling, alpha = alpha,
                              max_node = max_node, max_line = max_line,
                              plot_legend = True, labels = False,
                              node_color = node_color, edgecolors = edgecolors, hatch = hatch,
                              print_cbar = True, ncol= ncol, legend_loc = legend_loc,
                              plot_shape = plot_shape, marker = marker,
                              bbox_node = bbox_node, bbox_line = bbox_line,
                              fontsize = fontsize)
                else:
                    r.plotMap(linetype = linetype,  linestyle = linestyle, line_color =line_color,
                              objects = objects, nodes = nodes,
                              line_lim = line_lim, bus_lim = bus_lim, nodetype = cap_type,
                              fig = fig, ax = axes[i], mkr_scaling= mkr_scaling, alpha = alpha,
                              max_node = max_node, max_line = max_line,
                              plot_legend = False, labels = False,
                              node_color = node_color, edgecolors = edgecolors, hatch = hatch,
                              print_cbar = False, plot_shape = plot_shape, marker = marker,
                              bbox_node = bbox_node, bbox_line = bbox_line,
                              fontsize = fontsize)
                i += 1
        plt.tight_layout(pad = 0.0)
                
    def getSummary(self):
        
       summary = pd.DataFrame()
       
       summary['Overhead lines (GW)'] = self.getLineInv().sum()/1000
       summary['Pipelines (GW)'] = self.getLineInv(nodes = 'H2_NODES').sum()/30000
       summary['Hydrogen storage (GW)'] = self.getInv(objects = 'HYDROGEN_STORAGE', cap_type = 'power').sum(axis = 1,level = 0).sum()/30000
       summary['Battery storage (GW)'] = self.getInv(objects = 'BATTERY_STORAGE', cap_type = 'power').sum(axis = 1,level = 0).sum()/1000
       wind = self.getValueByType('prod', objects='ONSHORE_WIND_POWER_PLANTS').sum().sum(level = 0)
       solar = self.getValueByType('prod', objects='SOLAR_POWER_PLANTS').sum().sum(level = 0)
       prod = self.getValueByType('prod', objects='POWER_PLANTS').sum().sum(level = 0)
       summary['Renewable share (%)'] = ((wind + solar)/prod)*100
       summary['Total Emissions (tonne)'] = self.getProdByType('prod', 'CO2_coef', objects = 'PLANTS') .sum().sum(level = 0)/1000
       smr_en = self.getValueByType('prod', objects = ['SMR_PLANTS','SMR_CCS_PLANTS']).sum().sum(level = 0)*0.0333
       el_en = self.getValueByType('prod', objects = 'POWER_PLANTS').sum().sum(level = 0)
       pemel = self.getValueByType('prod', objects = ['PEMEL_PLANTS']).sum().sum(level = 0)
       pemel_en = pemel*0.0333
       pemel_l = pemel*51.3
       tot_en = (pemel_en + smr_en)*0.6 + el_en - pemel_l
       summary['Relative Emissions (tonne CO$_2$/MWh$_{el}$)'] = summary['Total Emissions (tonne)']/tot_en
       summary['Total Energy (MWh)'] = tot_en
       summary.index = [int(i*1000) for i in summary.index]
       
       return summary
   
    def getEnergyWaste(self):
        
        energy_waste = pd.DataFrame()
        
        load = self.getValueByType('Load', objects = 'EL_NODES').sum().sum(level = 0)
        wind_cur = self.getValueByType('cur', objects = 'ONSHORE_WIND_POWER_PLANTS').sum().sum(level = 0)
        wind_cur_share = (wind_cur/load)*100
        solar_cur = self.getValueByType('cur', objects = 'SOLAR_POWER_PLANTS').sum().sum(level = 0)
        solar_cur_share = (solar_cur/load)*100
        tot_cur = wind_cur_share + solar_cur_share
        energy_waste['Wind power cur (%)'] = wind_cur_share
        energy_waste['Solar power cur (%)'] = solar_cur_share
        energy_waste['Curtailment (%)'] = tot_cur
        rat = self.getValueByType('rat', objects = 'EL_NODES').sum().sum(level = 0)
        energy_waste['Rationing (%)'] = (tot_cur/load)*100
        energy_waste.index = [int(i*1000) for i in energy_waste.index]
        
        return energy_waste.round(2)
        
