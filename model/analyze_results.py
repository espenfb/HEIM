# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 12:57:34 2019

@author: espenfb
"""

import metaModel as mm
import pandas as pd
import numpy as np

time_data = {
'start_date': pd.Timestamp(year = 2015, month = 1, day = 1),
'end_date': pd.Timestamp(year = 2015, month = 12, day = 31),
'ref_date': pd.Timestamp(year = 2015, month = 1, day = 1)}
dirs = {
'data_dir' : "Data\\",
'ctrl_data_file' : 'ctrl_data.csv',
'res_dir' : ''}
meta_data = {'type': 'parameters',
             'param': 'CO2_cost',
             'range': np.arange(0.00,0.29,0.03)}

#res_dirs = ['NEEDS_new_lines_high_bat\\Meta_results\\',
#            'NEEDS_new_lines_high_bat\\Meta_results_10X\\',
#            'NEEDS_new_lines_high_bat\\Meta_results_100X\\']

res_dirs = ['NEEDS_h2trans\\Meta_results_double_line\\',
            'NEEDS_h2trans\\Meta_results_double_line_10X\\',
            'NEEDS_h2trans\\Meta_results_double_line_100X\\']

labels = ['Base Case','10x H$_2$ Demand', '100x H$_2$ Demand']
labels_sub = ['(a) Base case','(b) 10x H$_2$ Demand', '(c) 100x H$_2$ Demand']
k2t = 1000

res = []
for r_dir in res_dirs:
    
    dirs['res_dir'] = r_dir
    obj = mm.metaModel(time_data, dirs, meta_data)
    obj.loadRes()
    res.append(obj)
    
dirs['res_dir'] = 'NEEDS_h2trans\\Meta_results_noh2\\'
zh2 = mm.metaModel(time_data, dirs, meta_data)
zh2.loadRes()

import matplotlib.pyplot as plt    

plt.figure()
ax = plt.gca()
for r in res:
  h2_price = r.getH2Price().T
  h2_price.index = h2_price.index*k2t # go from $/kg to $/ton
  h2_price.plot(ax = ax)
  print(h2_price)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(labels, frameon=False)
plt.xlabel('CO$_{2}$ price [\$/ton]', fontsize = 12)
plt.ylabel('H$_{2}$ cost [\$/kg]', fontsize = 12)


fig,axes = plt.subplots(1,3, sharex = True)
for n,r in enumerate(res):
    ax =  axes[n]
    r.plotH2Source(plotType = 'area',ax = ax)
    if n == 0:
        ax.set_ylabel('Hydrogen source [kg]', fontsize = 12)
    if n == 1:
        ax.set_xlabel('CO$_{2}$ price [\$/ton]', fontsize = 12)
    if n == 0:
        ax.legend(frameon=False, bbox_to_anchor=(0., 1.15, 3., .102), loc='lower left',
           ncol=4, mode="expand", borderaxespad=0.)
    else:
        leg = ax.legend()
        leg.remove()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(labels_sub[n])
    ax.legend = None
    plt.tight_layout()
    

fig, axes = plt.subplots(1,3)
ze = zh2.getTotalEmissions()
for n,r in enumerate(res):
    eps = r.getPowerSystemEmissions()['Emissions from PS [CO2]'] - ze['Tot Emissions [ton CO2]']
    eng = r.getHydrogenNgEmissions()['Emissions from H2 [ton CO2]']
    h2d = r.getH2Source().sum()
    erps = eps/h2d
    erng = eng/h2d
    er = erps + erng
    axes[0].set_title('(a) Total')
    er.index = er.index*k2t
    er.plot(ax = axes[0])
    axes[1].set_title('(b) Natural Gas')
    erng.index = erng.index*k2t
    erng.plot(ax = axes[1])
    axes[2].set_title('(c) Electricity')
    erps.index = erps.index*k2t
    erps.plot(ax = axes[2])
for ax in axes:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
axes[1].set_xlabel('CO$_{2}$ price [\$/ton]', fontsize = 12)
axes[0].set_ylabel('Emissions [kg CO$_{2}$/ kg H$_{2}$]', fontsize = 12)
axes[0].legend(labels, frameon=False, bbox_to_anchor=(0., 1.15, 2., .102),
                loc='lower left', ncol=3, mode="expand", borderaxespad=0.)
plt.tight_layout()

plt.figure('Power price')
ax = plt.gca()
case = res[1]
pm,pstd = case.getPriceStats()
for i in pm.columns:
    plt.scatter(pm[i],pstd[i]) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend([int(i) for i in pm.columns*k2t], title = 'CO$_2$ price [\$/ton]', ncol = 2)
plt.xlabel('Price mean [$/MWh]', fontsize = 12)
plt.ylabel('Price std [$/MWh]', fontsize = 12)


res[0].res[0].plotMap(nodesize = 'Battery Power')


plt.figure('Bat and trans')
ax = plt.gca()
case = res[0]
bat = case.getBatteryInv()
bat.index = bat.index*k2t
line = case.getLineInv()
line.index = line.index*k2t
bat.plot(ax = ax)
line.plot(ax = ax)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlabel('CO$_{2}$ price [\$/ton]', fontsize = 12)
plt.ylabel('New Capacity [MW]', fontsize = 12)
plt.legend(['Battery','Transmission'], frameon=False)

case = res[0]
case.plotEnergySumByType(plotType = 'area')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlabel('CO$_{2}$ price [\$/ton]', fontsize = 12)
plt.ylabel('Energy [MWh]', fontsize = 12)

print(res[0].getElecOverSizing())

print(res[0].getH2StorageDur())