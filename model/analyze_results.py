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

res_dirs = ['NEEDS_fixed_hydrogen_load\\Meta_results\\',
            'NEEDS_fixed_hydrogen_load\\Meta_results_10X\\',
            'NEEDS_fixed_hydrogen_load\\Meta_results_100X\\']

#res_dirs = ['NEEDS_h2trans\\Meta_results_double_line\\',
#            'NEEDS_h2trans\\Meta_results_double_line_10X\\',
#            'NEEDS_h2trans\\Meta_results_double_line_100X\\']

#res_dirs = ['NEEDS2\\Meta_results\\',
#            'NEEDS\\Meta_results_10x\\',
#            'NEEDS\\Meta_results_100x_2\\']

#labels = ['Base Case','10x H$_2$ Demand','100x H$_2$ Demand']
labels_sub = ['(a) Base case','(b) 10x H$_2$ demand','(c) 100x H$_2$ demand']
labels = labels_sub
k2t = 1000

res = []
for r_dir in res_dirs:
    
    dirs['res_dir'] = r_dir
    obj = mm.metaModel(time_data, dirs, meta_data)
    obj.loadRes()
    res.append(obj)
    
dirs['res_dir'] = 'NEEDS\\Meta_results_noh2\\'
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
plt.xlabel('CO$_{2}$ price [\$/tonne]', fontsize = 12)
plt.ylabel('Hydrogen price [\$/kg]', fontsize = 12)


fig,axes = plt.subplots(1,3, sharex = True)
for n,r in enumerate(res):
    ax =  axes[n]
    r.plotH2Source(plotType = 'area',ax = ax)
    if n == 0:
        ax.set_ylabel('Hydrogen source [kg]', fontsize = 12)
    if n == 1:
        ax.set_xlabel('CO$_{2}$ price [\$/tonne]', fontsize = 12)
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


fig, ax = plt.subplots(1)
ze = zh2.getPowerSystemEmissions()
for n,r in enumerate(res):
    eps = r.getPowerSystemEmissions()['Emissions from PS [CO2]']/k2t
    eps.index = eps.index*k2t
    eps.plot(ax = ax)
ze_ton = ze/k2t
ze_ton.index = ze_ton.index*k2t
ze_ton.plot(ax = ax)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('CO$_{2}$ price [\$/tonne]', fontsize = 12)
ax.set_ylabel('Emissions [Tonnes CO$_{2}$]', fontsize = 12)
ax.legend(labels + ['No H$_2$ - Reference'], frameon=False,
                loc='upper right', ncol=1, borderaxespad=0.)
plt.tight_layout()   

fig, ax = plt.subplots(1)
ze = zh2.getPowerSystemEmissions()
for n,r in enumerate(res):
    eps = r.getPowerSystemEmissions()['Emissions from PS [CO2]'] - ze['Emissions from PS [CO2]']
    eng = r.getHydrogenNgEmissions()['Emissions from H2 [ton CO2]']
    #h2d = r.getH2Source().sum()
    erps = eps/k2t#/h2d
    erng = eng/k2t#/h2d
    er = erps + erng
    er.index = er.index*k2t
    er.plot(ax = ax)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('CO$_{2}$ price [\$/tonne]', fontsize = 12)
ax.set_ylabel('Emissions [Tonnes CO$_{2}$]', fontsize = 12)
ax.legend(labels, frameon=False, loc='upper right', ncol=1, borderaxespad=0.)
plt.tight_layout()   

fig, axes = plt.subplots(1,3)
for n,r in enumerate(res):
    eps = r.getPowerSystemEmissions()['Emissions from PS [CO2]'] - ze['Emissions from PS [CO2]']
    eng = r.getHydrogenNgEmissions()['Emissions from H2 [ton CO2]']
    #h2d = r.getH2Source().sum()
    erps = eps/k2t#/h2d
    erng = eng/k2t#/h2d
    er = erps + erng
    axes[0].set_title('(a) Total H$_2$')
    er.index = er.index*k2t
    er.plot(ax = axes[0])
    axes[1].set_title('(b) SMR')
    erng.index = erng.index*k2t
    erng.plot(ax = axes[1])
    axes[2].set_title('(c) Electrolysis')
    erps.index = erps.index*k2t
    erps.plot(ax = axes[2])
for ax in axes:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
axes[1].set_xlabel('CO$_{2}$ price [\$/tonne]', fontsize = 12)
#axes[0].set_ylabel('Emissions [kg CO$_{2}$/ kg H$_{2}$]', fontsize = 12)
axes[0].set_ylabel('Emissions [Tonnes CO$_{2}$]', fontsize = 12)
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
plt.legend([int(i) for i in pm.columns*k2t], title = 'CO$_2$ price [\$/tonne]', ncol = 2)
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
plt.xlabel('CO$_{2}$ price [\$/tonne]', fontsize = 12)
plt.ylabel('New Capacity [MW]', fontsize = 12)
plt.legend(['Battery','Transmission'], frameon=False)

case = res[0]
case.plotEnergySumByType(plotType = 'area')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlabel('CO$_{2}$ price [\$/tonne]', fontsize = 12)
plt.ylabel('Energy [MWh]', fontsize = 12)

print(res[0].getElecOverSizing())

print(res[0].getH2StorageDur())

c = ['C0','C1','C2','C3']
fig,ax = plt.subplots(1,1)
for n,r in enumerate(res):
    h2d = r.getH2Source().sum().T
    h2bysource = r.getH2Source()
    rel_h2 = (h2bysource/h2d)
    rel_h2 = rel_h2*100 # share to pcnt
    rel_h2.columns = rel_h2.columns*k2t # kg to ton
    (rel_h2.loc['Elec via storage'] + rel_h2.loc['Elec direct']).plot(ax = ax, color = c[n])
ax.legend(labels, ncol=1, frameon=False, fontsize = 10)
#ax.set_title('(a) H$_2$ from electrolysis', fontsize = 12)
ax.set_ylabel('Share hydrogen [%]', fontsize = 12)
ax.set_xlabel('CO$_{2}$ price [\$/tonne]', fontsize = 12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()


c = ['C0','C1','C2','C3']
fig,ax = plt.subplots(1,1)
for n,r in enumerate(res):
    h2d = r.getH2Source().sum().T
    h2bysource = r.getH2Source()
    rel_h2 = (h2bysource/h2d)
    rel_h2 = rel_h2*100 # share to pcnt
    rel_h2.columns = rel_h2.columns*k2t # kg to ton
    #ax.plot([0,0],[40,40], color = 'w')
    rel_h2.loc['Natural Gas'][rel_h2.loc['Natural Gas']>0.0].plot(ax = ax, color = c[n])
    rel_h2.loc['Natural Gas CCS'][rel_h2.loc['Natural Gas CCS']>0.0].plot(ax = ax, color = c[n], linestyle = '--')
#ax.legend(['Base case: SMR','Base case: SMR with CCS','10x: SMR','10x: SMR with CCS', '100x: SMR','100x: SMR with CCS'],
#    ncol=1, frameon=False, fontsize = 10)
ax.legend(labels, ncol=1, frameon=False, fontsize = 10)
#ax.set_title('(b) H$_2$ from natural gas (SMR)', fontsize = 12)
ax.set_ylabel('Share hydrogen [%]', fontsize = 12)
ax.set_xlabel('CO$_{2}$ price [\$/tonne]', fontsize = 12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()



c = ['C0','C1','C2','C3']
fig,axes = plt.subplots(1,2)
for n,r in enumerate(res):
    h2d = r.getH2Source().sum().T
    h2bysource = r.getH2Source()
    rel_h2 = (h2bysource/h2d)
    rel_h2 = rel_h2*100 # share to pcnt
    rel_h2.columns = rel_h2.columns*k2t # kg to ton
    (rel_h2.loc['Elec via storage'] + rel_h2.loc['Elec direct']).plot(ax = axes[0], color = c[n])
    rel_h2.loc['Natural Gas'][rel_h2.loc['Natural Gas']>0.0].plot(ax = axes[1], color = c[n])
    rel_h2.loc['Natural Gas CCS'][rel_h2.loc['Natural Gas CCS']>0.0].plot(ax = axes[1], color = c[n], linestyle = '--')
axes[0].legend(labels, frameon=False, bbox_to_anchor=(0., 1.10, 1.5, .102),
                loc='lower left', ncol=3, mode="expand", borderaxespad=0., fontsize = 12)
axes[0].set_title('(a) H$_2$ from electrolysis', fontsize = 12)
axes[0].set_ylabel('Share hydrogen [%]', fontsize = 12)
axes[0].set_xlabel('CO$_{2}$ price [\$/tonne]', fontsize = 12)
axes[1].set_title('(b) H$_2$ from natural gas (SMR)', fontsize = 12)
axes[1].set_ylabel('Share hydrogen [%]', fontsize = 12)
axes[1].set_xlabel('CO$_{2}$ price [\$/tonne]', fontsize = 12)
axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
plt.tight_layout()

c = ['C0','C1','C2','C3']
fig,ax = plt.subplots(1,1)
for n,r in enumerate(res):
#    ax.plot([0,0],[0,0], color = 'w')
    s = r.getH2StorageCap()
    s.columns = s.columns*k2t
    d_all = r.getH2Source()
    #d = d_all.loc['Elec direct'] + d_all.loc['Elec via storage']
    d = d_all.loc['Elec via storage']
    d.index = d.index*k2t
    s_hrs = (s.sum()/(d.sum()/8760))
    s_hrs.plot(ax = ax, color = c[n])
#    b_dur = r.getStorageDuration().mean()
#    b_dur.index = b_dur.index*k2t
#    b_dur.plot(ax = ax, color = c[n], linestyle = '--')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Storage [h]', fontsize = 12)
ax.set_xlabel('CO$_{2}$ price [\$/tonne]', fontsize = 12)
ax.legend(labels,frameon=False, fontsize = 10)
#ax.legend([labels[0],'Electrolysis','Battery',labels[1],'Electrolysis','Battery',labels[2],'Electrolysis','Battery'],
#    ncol=3, frameon=False, fontsize = 10, bbox_to_anchor=(0., 1.00, 2., .102),
#                loc='lower left')


c = ['C0','C1','C2','C3']
fig,ax = plt.subplots(1,1)
for n,r in enumerate(res):
    s = r.getH2StorageCap()
    s.columns = s.columns*k2t
    energy_rate = r.res[0].data.hydrogen_plant_char['Energy rate [MWh/kg]'].sum()
    e = r.getElecCap()/energy_rate 
    e.columns = e.columns*k2t
    s_hrs = (s.sum()/e.sum())
    s_hrs.plot(ax = ax, color = c[n])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Duration [h]', fontsize = 12)
ax.set_xlabel('CO$_{2}$ price [\$/tonne]', fontsize = 12)
ax.legend(labels,frameon=False, fontsize = 10, ncol=3,
          bbox_to_anchor=(0., 1.00, 2., .102), loc='lower left')

for n,r in enumerate(res):
    b_dur = r.getStorageDuration().mean()
    b_dur.index = b_dur.index*k2t
    b_dur.plot(ax = ax, color = c[n], linestyle = '--')


    
c = ['C0','C1','C2','C3']
fig,ax = plt.subplots(1,1)
for n,r in enumerate(res):
    ax.plot([0,0],[40,40], color = 'w')
    elec = r.getElecUtilization()*100
    elec.columns = elec.columns*k2t
    elec.mean().plot(color = c[n], linestyle = '--')
    bat = r.getBatteryUtilization()*100
    bat.columns = bat.columns*k2t
    bat.mean().plot(color = c[n])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend([labels[0],'Electrolysis','Battery',labels[1],'Electrolysis','Battery',labels[2],'Electrolysis','Battery'],
    ncol=3, frameon=False, fontsize = 8)
ax.set_ylabel('Capacity utilization [%]', fontsize = 12)
ax.set_xlabel('CO$_{2}$ price [\$/tonne]', fontsize = 12)

c = ['C0','C1','C2','C3']
fig,ax = plt.subplots(1,1)
for n,r in enumerate(res):
    ap = r.getAverageH2PowerPrice()*100
    ap.columns = ap.columns*k2t
    ap.mean().plot(color = c[n])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Electricity price [% of average]', fontsize = 12)
ax.set_xlabel('CO$_{2}$ price [\$/tonne]', fontsize = 12)
ax.legend(labels,frameon=False, fontsize = 10)