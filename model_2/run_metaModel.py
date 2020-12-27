# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:55:31 2019

@author: espenfb
"""

import metaModel as mm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import metaRes as mr
 
idx = pd.IndexSlice

time_data = {
'start_date': pd.Timestamp(year = 2015, month = 1, day = 1),
'end_date': pd.Timestamp(year = 2016, month = 1, day = 1),
'ref_date': pd.Timestamp(year = 2015, month = 1, day = 1)}

dirs = {
'data_dir' : "Data\\",
'ctrl_data_file' : 'ctrl_data.csv',
#'res_dir' : 'NEEDS2\\Meta_results_100x\\'}
'res_dir' : 'Result_3\\Year\\'}

meta_data = {
'param': 'CO2_cost',
 'index': 'None',
 'kind': 'absolute',
 'range': np.arange(0.00,0.29,0.03)} 

obj = mm.metaModel(time_data, dirs, meta_data)

obj.runMetaModel()


# Examples of result plotting:
# see file: savedRes.py for more information

plt.close('all') # close all plots

# Plot meta model results:


pt = 'bar'

if True: obj.meta_res = mr.metaRes(obj.res_dir, obj.meta_data, data = obj.model.data)

obj.meta_res.plotValueByType('prod', objects = 'POWER_PLANTS', kind = 'area',
                    lower_lim = 1000000)
obj.meta_res.plotValueByType('prod', objects = 'H2_PLANTS', kind = 'bar', xscale = 1000)

# Plot results for individual scenario:

res = obj.meta_res.res[-1] # results for the first parameter run of meta model


obj.meta_res.plotValueByType('cur', kind = 'bar', base_val = 'Load',
                    base_obj = 'EL_NODES', xscale=1000)

obj.meta_res.plotInvByType(objects = 'STORAGE', xscale = 1000,
                  lower_lim = 0.0, conH2power = True)

#res.plotEnergyByType()
res.getValueByType('prod',
                   objects = 'POWER_PLANTS',
                   lower_lim = 100).plot(kind = 'area',
                                        cmap = 'tab20c')

#res.plotInvByBus(lower_limit = 100.0)

res.plotValue('storage', objects= 'BATTERY_STORAGE')
res.plotValue('storage', objects= 'HYDROGEN_STORAGE')

mkr_scale = 500

res.plotMap(linetype = 'Cap', objects = 'BATTERY_STORAGE', mkr_scaling= 100, rel_lines = True)
res.plotMap(linetype = 'Cap', nodes = 'H2_NODES', 
            objects = 'HYDROGEN_STORAGE',
            nodetype= 'energy', mkr_scaling= mkr_scale, line_lim = 1000)

res.plotMap(objects = 'PEMEL_PLANTS', nodetype = 'power', mkr_scaling= mkr_scale)

res.plotMap(objects = 'ONSHORE_WIND_POWER_PLANTS', nodetype = 'power', mkr_scaling= mkr_scale)
res.plotMap(objects = 'SOLAR_POWER_PLANTS', nodetype = 'power', mkr_scaling= mkr_scale)
res.plotMap(objects = 'SMR_PLANTS', nodetype = 'power', mkr_scaling= mkr_scale)
res.plotMap(objects = 'SMR_CCS_PLANTS', nodetype = 'power', mkr_scaling= mkr_scale)

total_load = res.opr_res.loc[idx[:],idx[res.EL_NODES,'Load']].sum().sum()
total_rat = res.opr_res.loc[idx[:],idx[res.EL_NODES,'rat']].sum().sum()
print('Total rationing: ','%.1f' % total_rat, ' MWh (%.2f percent of total load)' % ((total_rat/total_load)*100))
total_h2_load = res.opr_res.loc[idx[:],idx[res.H2_NODES,'Load']].sum().sum()
total_h2_rat = res.opr_res.loc[idx[:],idx[res.H2_NODES,'rat']].sum().sum()
print('Total H2 rationing: ','%.1f' % total_h2_rat, ' MWh (%.2f percent of total load)' % ((total_rat/total_load)*100))
wind_cur = res.opr_res.loc[idx[:],idx[res.WIND_POWER_PLANTS,'cur']].sum().sum()
print('Total wind power curtailment: ','%.1f' % wind_cur, ' MWh (%.2f percent of total load)' % ((wind_cur/total_load)*100))
solar_cur = res.opr_res.loc[idx[:],idx[res.SOLAR_POWER_PLANTS,'cur']].sum().sum()
print('Total solar power curtailment: ','%.1f' % solar_cur, ' MWh (%.2f percent of total load)' % ((solar_cur/total_load)*100))
