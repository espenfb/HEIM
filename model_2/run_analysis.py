# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:25:02 2020

@author: espenfb
"""

import resultAnalysis as ra
import systemData as sd
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd
idx = pd.IndexSlice

# font = {'family' : 'serif',
#         'weight' : 'normal'}
# import matplotlib
# matplotlib.rc('font', **font)

import matplotlib.pylab as pylab
fs = 26
params = {'legend.fontsize': fs-1,
          'legend.title_fontsize':fs,
          #'figure.figsize': (15, 4.5),
         'axes.labelsize': fs,
         'axes.titlesize':fs,
         'xtick.labelsize':fs,
         'ytick.labelsize':fs,
         'hatch.linewidth': 1.0,
         'axes.axisbelow': True,
         'font.family': 'serif',
         'font.weight': 'normal',
         'font.size': fs}
pylab.rcParams.update(params)

data_dirs = {
'data_dir' : "Data\\",
'ctrl_data_file' : 'ctrl_data.csv'}

data = sd.systemData(data_dirs)


meta_data = {
'param': 'CO2_cost',
'index': 'None',
'kind': 'absolute',
'range': np.arange(0.00,0.29,0.03)} 

sep = '\\'
folder = 'Result_2tier'

# res_dirs = {'No hydrogen': 'Year_no', '(a) base case': 'Year_2',
#             '(b) 10x': 'Year10x'}#, '(c) 100x': 'Year100x'}

#res_dirs = {'(a) base case': 'Year', '(b) 10x': 'Year10x', '(c) 50x': 'Year50x'}
res_dirs = {'no H$_2$': 'Year_no','(a) base case': 'Year', '(b) 10X H$_2$': 'Year10x_p20', '(c) 50X H$_2$': 'Year50x_p40'}

res_dirs = {k: folder + sep + v + sep for k,v in res_dirs.items()}

res_dirs_h2 = copy.copy(res_dirs)
res_dirs_h2.pop('no H$_2$')

if True:
    res = ra.resultAnalysis(res_dirs, meta_data, sep = sep, data = data)
    
    res_h2 = ra.resultAnalysis(res_dirs_h2, meta_data, sep = sep, data = data)

h2_source =  res.getValueByType('prod', objects = 'H2_PLANTS', aggr = 'sum')


color_dict = {'Nuclear': 'royalblue', 'Coal':'grey', 'CC Gas': 'indianred', 'CCS Gas': 'darkred',
              'CT Gas': 'saddlebrown','Wind': 'seagreen', 'Solar': 'orange'} # Onshore Wind changed to Wind

color_dict_h2 = {'PEMEL': 'seagreen', 'SMR':'grey', 'SMR CCS': 'royalblue'}

color_tot = {**color_dict, **color_dict_h2}
color_tot['Coal'] = 'k'

color_case = {'no H$_2$': 'lightgrey','(a) base case': 'k', '(b) 10X H$_2$': 'steelblue', '(c) 50X H$_2$': 'lightsteelblue'}


if True:
    res_h2.plotValueByType('prod', objects = 'H2_PLANTS', aggr = 'sum', stacked = True,
                        xlabel = 'CO$_2$ price (\$/tonne)', ylabel = 'Hydrogen source (%)',
                        xscale = 1000, yscale = 100, width = 0.8,
                        color_dict = color_dict_h2, bbox =(0.35,0.90), leg_loc = 'lower left')
    
    res_h2.plotValueByType('prod', objects = 'POWER_PLANTS', aggr = 'sum', stacked = True,
                        xlabel = 'CO$_2$ price (\$/tonne)', ylabel = 'Electricity source (%)',
                        xscale = 1000, yscale = 100, width = 0.8, lower_lim = 0.5, leg_col = 7,
                        color_dict = color_dict, bbox =(0.15,0.90), leg_loc = 'lower left')
    
    # res.plotValueByType('prod', objects = 'POWER_PLANTS', aggr = 'sum', stacked = True,
    #                     xlabel = 'CO$_2$ price (\$/tonne)', ylabel = 'Electricity source (%)',
    #                     xscale = 1000, width = 0.8, lower_lim = 5000000, leg_col = 7,
    #                     color_dict = color_dict, relative = False)

res_h2.plotProdByType('prod', 'CO2_coef', objects = 'PLANTS', aggr = 'sum',
                   kind = 'bar', stacked = True, relative = False,
                    xlabel = 'CO$_2$ price (\$/tonne)', leg_col = 1,leg_loc = 'upper right',
                    ylabel = 'CO$_2$ emissions (million tonnes)', bar_labels = True,
                    xscale = 1000, yscale = 1E-9, lower_lim = 0.1,
                    color_dict = color_tot, width = 0.8,
                    sharey = False, first_base = True)

# res.plotProdByType('prod', 'CO2_coef', objects = 'POWER_PLANTS', aggr = 'sum',
#                    kind = 'line', stacked = False, relative = False,
#                     xlabel = 'CO$_2$ price (\$/tonne)',
#                     ylabel = 'CO$_2$ emissions (tonnes)', bar_labels = False,
#                     xscale = 1000, yscale = 0.001, cmap = 'tab20c',
#                     lower_lim = 1, sharey = True,sum_types = True, plot_type = 'single', marker = '.')

e = res.getProdByType('prod', 'CO2_coef', objects = 'PLANTS', aggr = 'sum', relative = False).sum(axis = 1, level = 0)
de = pd.DataFrame()
for i in e.columns:
    de[i] = e[i] - e['no H$_2$']
h2 = res_h2.getValueByType('prod', objects = 'H2_PLANTS', aggr = 'sum', relative = False).sum(axis = 1, level = 0)
re = (de/h2).dropna(axis = 1)
re.index = h2.index

res.singlePlot(re,kind = 'line', marker = 'o', leg_loc = 'upper right',
              linewidth = 3.0, xscale = 1000,
              xlabel = 'CO$_2$ price (\$/tonne)',
              ylabel = 'Relative  emissions \n(kg CO$_2$/kg H$_2$)', leg_at_fig = True, color_dict = color_case)

g = res.getValueByType('prod', objects = 'CC_GAS_POWER_PLANTS', aggr = 'sum', relative = False).sum(axis = 1, level = 0)
dg = pd.DataFrame()
for i in g.columns:
    dg[i] = (g[i] - g['no H$_2$'])/g['no H$_2$']
dg.max()    
dg.min() 
    
f,a = plt.subplots(1,3, sharex=False, sharey=False,
                           figsize=(5*3, 5*1))

res_h2.plotValue('energyBalance', objects = 'H2_NODES', by_node=True,
              aggr = 'mean', plot_type = 'single', kind = 'line', marker = 'o',
              sum_types = 'mean', unstack = True, leg_loc = 'upper left',
              linewidth = 3.0, yscale = 1, xscale = 1000, color_dict = color_case,
              xlabel = 'CO$_2$ price (\$/tonne)',
              ylabel = 'Hydrogen price (\$/kg)', leg_at_fig = False, no_leg = True, ax2 = a[2], text = '3')

res.plotValue('energyBalance', objects = 'EL_NODES', by_node=True,
              aggr = 'mean', plot_type = 'single', kind = 'line', marker = 'o',
              sum_types = 'mean', unstack = True, leg_loc = 'upper left',
              linewidth = 3.0, yscale = 1, xscale = 1000, color_dict = color_case, leg_at_fig = False,  no_leg = True,
              xlabel = 'CO$_2$ price (\$/tonne)',
              ylabel = 'Electricity price (\$/MWh)', ax2 = a[0], text = '1')

res.plotPriceIQR( kind = 'line', leg_loc = 'upper left', marker = 'o',
              linewidth = 3.0, yscale = 1, xscale = 1000, color_dict = color_case,
              xlabel = 'CO$_2$ price (\$/tonne)',
              ylabel = 'El. price IQR (\$/MWh)',  no_leg = True, ax2 = a[1], text = '2')

for i in a:
    leg = i.get_legend()
    if leg: leg.remove()
labels = a[0].get_legend_handles_labels()[1]
f.legend(labels,loc = 'lower left', ncol = 4, frameon=False, bbox_to_anchor=(0.25,0.89))

# res_h2.plotWeightedValue('energyBalance', 'prod', val_objects='EL_NODES', marker = 'o',
#                       weight_objects='PEMEL_PLANTS', weight_coeff = 'Conv_rate',
#                       kind = 'line', leg_col = 3, leg_loc = 'upper center',
#                       linewidth = 3.0, yscale = 100, xscale = 1000, color_dict = color_case,
#                       xlabel = 'CO$_2$ price (\$/tonne)',
#                       ylabel = 'PEMEL el. price (% of average)', xfilter= idx[1:])

# res.plotSummary(by_col = 'Renewable share (%)', kind = 'line',
#                 linewidth = 3.0,  marker = 'o', color_dict = color_case,
#                 leg_loc = 'upper left', leg_at_fig = False)

# res.plotSummary(kind = 'line', xlabel = 'CO$_2$ price (\$/tonne)',
#                 linewidth = 3.0,  marker = 'o', color_dict = color_case,
#                 leg_loc = 'lower right', leg_at_fig = False)

# res.plotStorageDuration(objects = 'BATTERY_STORAGE',
#               plot_type = 'single', kind = 'line', marker = 'o',
#               leg_loc = 'upper left',
#               linewidth = 3.0, yscale = 1, xscale = 1000, cmap = 'tab20c', leg_at_fig = False,
#               xlabel = 'CO$_2$ price (\$/tonne)',
#               ylabel = 'Battery storage duration (h)')

# res.plotStorageDuration(objects = 'HYDROGEN_STORAGE',
#               plot_type = 'single', kind = 'line', marker = 'o',
#               leg_loc = 'upper left',
#               linewidth = 3.0, yscale = 1, xscale = 1000, cmap = 'tab20c', leg_at_fig = False,
#               xlabel = 'CO$_2$ price (\$/tonne)',
#               ylabel = 'H$_2$ storage duration (h)')

# res.plotInv(objects = ['ONSHORE_WIND_POWER_PLANTS', 'SOLAR_POWER_PLANTS'],
#            kind = 'line', marker = 'o', leg_loc = 'upper left',
#               linewidth = 3.0, yscale = 0.001, xscale = 1000, color_dict = color_case,
#               xlabel = 'CO$_2$ price (\$/tonne)',
#               ylabel = 'Renewable capacity (GW)', leg_at_fig = False)

# res.plotInv(objects = ['PEMEL_PLANTS'],
#            kind = 'line', marker = 'o', leg_loc = 'upper left',
#               linewidth = 3.0, yscale = 1/30000, xscale = 1000, color_dict = color_case,
#               xlabel = 'CO$_2$ price (\$/tonne)',
#               ylabel = 'PEMEL capacity (GW)', leg_at_fig = False)

# res.plotInv(objects = ['HYDROGEN_STORAGE'],
#            kind = 'line', marker = 'o', leg_loc = 'upper left',
#               linewidth = 3.0, yscale = 1/30000, xscale = 1000, color_dict = color_case,
#               xlabel = 'CO$_2$ price (\$/tonne)',
#               ylabel = 'Capacity (GW)', leg_at_fig = False)

f,a = plt.subplots(1,3, sharex=False, sharey=False,
                           figsize=(5*3, 5*1))
res.plotSummaryType( 'Renewable share (%)', kind = 'line', xlabel = 'CO$_2$ price (\$/tonne)',
                linewidth = 3.0,  marker = 'o', color_dict = color_case,
                leg_loc = 'lower right', leg_at_fig = False, no_leg = True, ax2 = a[0], text = '1')

res.plotSummaryType( 'Overhead lines (GW)', kind = 'line', xlabel = 'CO$_2$ price (\$/tonne)',
                linewidth = 3.0,  marker = 'o', color_dict = color_case,
                leg_loc = 'lower right', leg_at_fig = False, no_leg = True, ax2 = a[2], text = '3')

res.plotInv(objects = ['BATTERY_STORAGE'],
           kind = 'line', marker = 'o', leg_loc = 'upper left',
              linewidth = 3.0, yscale = 1/1000, xscale = 1000, color_dict = color_case,
              xlabel = 'CO$_2$ price (\$/tonne)',
              ylabel = 'Battery capacity (GW)', leg_at_fig = False, ax2 = a[1], text = '2')

for i in a:
    leg = i.get_legend()
    if leg: leg.remove()
labels = a[0].get_legend_handles_labels()[1]
f.legend(labels,loc = 'lower left', ncol = 4, frameon=False, bbox_to_anchor=(0.25,0.89))

# res.plotInv(objects = ['PEMEL_PLANTS'],
#            kind = 'line', marker = 'o', leg_loc = 'upper left',
#               linewidth = 3.0, yscale = 1/30000, xscale = 1000, color_dict = color_case,
#               xlabel = 'CO$_2$ price (\$/tonne)',
#               ylabel = 'PEMEL capacity (GW)', leg_at_fig = False)

import matplotlib
#a = res.analysis['(a) base case']
a = res.analysis['(a) base case']
#b = res.analysis['(b) 10x']
#c = res.analysis['(c) 50x']

h2 = a.getInvByType(objects = ['PEMEL_PLANTS','SMR_PLANTS','SMR_CCS_PLANTS','HYDROGEN_STORAGE']).sum(level = 0,axis = 1)*(1/30000)
h2 = h2.rename({'H$_2$': 'H$_2$ Storage'})

h2s = a.getInvByType(objects = ['HYDROGEN_STORAGE'], cap_type = 'energy').sum(level = 0,axis = 1)*(1/30000)
h2s = h2s.rename({'H$_2$': 'H$_2$ Storage (energy)'})

el = a.getInvByType(objects = ['ONSHORE_WIND_POWER_PLANTS','SOLAR_POWER_PLANTS','BATTERY_STORAGE']).sum(level = 0,axis = 1)*(1/1000)
bat = a.getInvByType(objects = ['BATTERY_STORAGE'], cap_type = 'energy').sum(level = 0,axis = 1)*(1/1000)

# f,a = plt.subplots(2,1, sharex=False, sharey=False,
#                            figsize=(5*2, 5*1))
f = plt.figure()

a_01 = f.add_subplot(211, sharex=None, sharey=None)

res.singlePlot(el.T,kind = 'line', marker = 'o', leg_loc = 'lower left',
              linewidth = 3.0, xscale = 1000,
              xlabel = 'CO$_2$ price (\$/tonne)',leg_col = 3, sharex=False, sharey=False,
              ylabel = 'Capacity (GW)', leg_at_fig = False,
              color_dict = color_dict, ax2 = a_01 , bbox =(0.05,0.95), text = '1', zorder = 2)
nticks = 5 
a_01.set_ylim(ymin = 0, ymax= 100)
a_01.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
a_01.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))

a_11 = f.add_subplot(212, sharex=None, sharey=None)

res.singlePlot(h2.T,kind = 'line', marker = 'o', leg_loc = 'lower left',
              linewidth = 3.0, xscale = 1000,
              xlabel = 'CO$_2$ price (\$/tonne)',leg_col = 4, sharex=False, sharey=False,
              ylabel = 'Capacity (GW)', leg_at_fig = False,
              color_dict = color_dict_h2, ax2 = a_11, bbox =(-0.1,0.95), text = '2', zorder = 2)

a_11.set_ylim(ymin = 0, ymax= 4)
a_11.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
a_11.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))

a_02 = a_01.twinx()
res.singlePlot(bat.T,kind = 'line', marker = 'o',
              linewidth = 3.0, xscale = 1000,
              xlabel = 'CO$_2$ price (\$/tonne)',
              ylabel = 'Capacity (GWh)', no_leg = True, sharex=False, sharey=False,
              color_dict = color_dict, ax2 = a_02 ,
              linestyle = '--', zorder = 1)

a_02.set_ylim(ymin = 0, ymax= 200)
a_02.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
a_02.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
a_02.spines['right'].set_visible(True)
a_02.grid(False)
a_02.set_xlabel('CO$_2$ price (\$/tonne)')


a_12 = a_11.twinx()
res.singlePlot(h2s.T,kind = 'line', marker = 'o',
              linewidth = 3.0, xscale = 1000, no_leg = True, sharex=False, sharey=False,
              ylabel = 'Capacity (GWh)',
              color_dict = color_dict_h2, ax2 = a_12,
              linestyle = '--', zorder = 1)

a_12.set_xlabel('CO$_2$ price (\$/tonne)')
a_12.set_ylim(ymin = 0, ymax= 100)
a_12.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
a_12.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
a_12.spines['right'].set_visible(True)
a_12.grid(False)

[t.set_visible(True) for t in a_02.get_xticklabels()]

plt.subplots_adjust(wspace=0.8, hspace=0.8)
plt.show()


if True:
    a.plotMapSelection([0.00,0.09,0.27], objects = 'HYDROGEN_STORAGE',
                       nodes = 'H2_NODES',legend_loc='lower center', linetype = 'Cap',
                       title_prefix= ' CO$_2$ price: ', line_lim = 1000,mkr_scaling= 1000)
    
    selection = [0.03,0.12,0.27]
    mkr = 2000
    
    fig,axes = plt.subplots(1,len(selection),sharex=True, sharey=True)
    a.plotMapSelection(selection,
                       linetype = 'Cap', linestyle = 'width', line_color = 'k',
                       nodes = 'H2_NODES',
                       fig = fig, axes = axes,
                       legend_loc='lower left', bbox_line = (0.63,0.015),
                       title_prefix= '', line_lim = 1000, plot_shape = False)
    a.plotMapSelection(selection, objects = 'HYDROGEN_STORAGE', nodes = 'H2_NODES',
                       fig = fig, axes = axes,
                       node_color ='grey',
                       marker = 's', alpha = 1.0,
                       legend_loc='lower left', bbox_node = (0.295,0.015),
                       title_prefix= '', mkr_scaling= mkr, plot_shape = False)
    a.plotMapSelection(selection, objects = 'ONSHORE_WIND_POWER_PLANTS', nodes = 'H2_NODES', cap_type = 'power',
                       fig = fig, axes = axes,
                       edgecolors ='darkgreen', node_color ='None',
                       alpha = 1.0,
                       legend_loc='lower left', bbox_node = (0.-0.01,0.015),
                       title_prefix= '', mkr_scaling= mkr,
                       plot_shape = False)
    a.plotMapSelection(selection, objects = 'ONSHORE_WIND_POWER_PLANTS', nodes = 'H2_NODES', cap_type = 'power',
                       fig = fig, axes = axes,
                       node_color ='darkgreen',
                       alpha = 0.4,
                       legend_loc='lower left', bbox_node = (-0.01,0.015),
                       title_prefix= '', mkr_scaling= mkr,
                       plot_shape = True)

    
    
    fig,axes = plt.subplots(1,len(selection),sharex=True, sharey=True)
    a.plotMapSelection(selection,
                       linetype = 'Cap', linestyle = 'width', line_color = 'k',
                       nodes = 'EL_NODES',
                       fig = fig, axes = axes,
                       legend_loc='lower left', bbox_line = (0.63,0.015),
                       title_prefix= '', line_lim = 1000, plot_shape = False)
    a.plotMapSelection(selection, objects = 'BATTERY_STORAGE', nodes = 'EL_NODES',
                       fig = fig, axes = axes,
                       node_color ='grey', marker = 's', alpha = 1.0,
                       legend_loc='lower left', bbox_node = (0.295,0.015),
                       title_prefix= '', mkr_scaling= mkr, plot_shape = False)
    a.plotMapSelection(selection, objects = 'SOLAR_POWER_PLANTS', nodes = 'EL_NODES', cap_type = 'power',
                       fig = fig, axes = axes,
                       edgecolors ='darkorange', node_color ='None', alpha = 1.0,
                       legend_loc='lower left', bbox_node = (-0.01,0.015),
                       title_prefix= '', mkr_scaling= mkr,
                       plot_shape = False)
    a.plotMapSelection(selection, objects = 'SOLAR_POWER_PLANTS', nodes = 'EL_NODES', cap_type = 'power',
                       fig = fig, axes = axes,
                       node_color ='darkorange', alpha = 0.4,
                       legend_loc='lower left', bbox_node = (-0.01,0.015),
                       title_prefix= '', mkr_scaling= mkr,
                       plot_shape = True)

