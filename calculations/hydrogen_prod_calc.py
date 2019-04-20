# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:55:41 2019

@author: espenfb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


price_folder = '../prices/Nordpool/'

#prefix = 'RTM_'#'DAM_'
#prefix_list = ['RTM', 'DAM']
prefix_list = ['SPOT', 'REG']
only_load_zone = False
include_tariff = False

NO_zone_map = {'NO1': 'Oslo', 'NO2':'Kr.sand', 'NO3':'Tr.heim', 'NO4':'Tromsø',
               'NO5': 'Bergen'}

total_cost = {}
total_cost_w_tartiff = {}
    
for prefix in prefix_list:
    year = 2017
    
    filename = prefix + '_'  + str(year) + '.csv'
    file = price_folder + filename
    
    if prefix == 'SPOT':
        header = 0
    elif prefix == 'REG':
        header = [0,1]
    else:
        header = 0
    
    prices = pd.read_csv(file, index_col = 0, header = header)
    
    if only_load_zone:
        load_zone_col_indx = [('LZ' in i) for i in prices.columns]
        load_zone_col = prices.columns[load_zone_col_indx]
        lz_prices = prices[load_zone_col]
    elif prefix == 'SPOT':
        lz_prices = prices
        load_zone_col = prices.columns
    elif prefix == 'REG':
        lz_prices = prices
        load_zone_col = prices.columns.levels[0]
    
    conv_h2_nm3 = 4.66 #kW/Nm^3
    density = 0.0899 #kg/Nm^3
    kW2MW = 0.001
    MW2kW = 1000
    
    # Guessing prices are in $/MWh
    
    ## PARAMETERS ##
    hydrogen_demand = 50000 # kg/day
    day_pr_year = 365 # day/year
    month_in_year = 12 # month/year
    hours_pr_day = 24 # h/day
    
    ## ENERGY CONSUMPTION ##
    
    energy_consumption_kg = (conv_h2_nm3/density)*kW2MW
    
    energy_consumption_day = energy_consumption_kg*hydrogen_demand
    
    energy_consumption_year = energy_consumption_day*day_pr_year
    
    ## ENERGY COSTS ##
    
    mean_lz_prices = lz_prices.mean()
    
    energy_cost_year = pd.Series()
    for zone in load_zone_col:

        if prefix == 'SPOT':
            mean_price = lz_prices[zone].mean()
            zone_name = zone
            if zone == 'SYS' or zone == 'Molde':
                continue
        elif prefix == 'REG':
            mean_price = lz_prices[zone]['Down'].mean()
            if zone in NO_zone_map.keys():
                zone_name = NO_zone_map[zone]
            else:
                zone_name = zone
        
        
        if only_load_zone:
            index = zone_name.replace('LZ_','')
        else:
            index = zone_name
            
        energy_cost_year.loc[index] = energy_consumption_year*mean_price
        

    
    zones = energy_cost_year.index.tolist() 
    
    energy_cost_year.plot(kind = 'bar')
    
    ## ELEC CAPACITY ##
    elec_capacity = conv_h2_nm3*kW2MW*(hydrogen_demand/density)/hours_pr_day
    
    ## TARIFF COST ##
    if include_tariff:
        utilities = pd.read_csv('../utilities.csv', index_col = 0, header = 0,
                                skipinitialspace = True)
        loadzone2utility = {'AEN':['Oncor'], 'CPS' : ['AEP_c'],
                            'HOUSTON': ['CenterPoint','TNMP'],
                            'LCRA': ['AEP_c','Oncor','AEP_n'],
                            'NORTH': ['Oncor'], 'RAYBN': ['Oncor'],
                            'SOUTH': ['AEP_c'], 'WEST':['Oncor','AEP_n','TNMP']}
        
        print(elec_capacity)
        
        tariff = pd.Series()
        for i in zones:
            for j in loadzone2utility[i]:
                firm = utilities.loc[j,'fixed']
                power = utilities.loc[j,'var_p']*elec_capacity*MW2kW*month_in_year
                energy = utilities.loc[j,'var_e']*energy_consumption_year*MW2kW
                tariff.loc[i + '+' + j] = firm + power + energy
            
    ## INV COST ##
            
    elec_cost = 69469 # 2015EUR/MW
    EUR2USD = 1.0672 # USD/EUR
    inv_cost_elec = elec_cost*elec_capacity*EUR2USD
    
    ## TOTAL COST ##
    
    total_cost[prefix] = pd.DataFrame()
    total_cost_w_tartiff[prefix] = pd.DataFrame()
    for i in zones:
        total_cost[prefix].loc[i, 'investment'] = inv_cost_elec/(hydrogen_demand*day_pr_year)
        total_cost[prefix].loc[i, 'energy'] = energy_cost_year.loc[i]/(hydrogen_demand*day_pr_year)
        if include_tariff:
            for j in loadzone2utility[i]:
                total_cost_w_tartiff[prefix].loc[i + '+' + j, 'investment'] = \
                                            inv_cost_elec/(hydrogen_demand*day_pr_year)
                total_cost_w_tartiff[prefix].loc[i + '+' + j, 'energy'] = \
                                energy_cost_year.loc[i]/(hydrogen_demand*day_pr_year)
                total_cost_w_tartiff[prefix].loc[i + '+' + j, 'tariff'] = \
                                tariff.loc[i + '+' + j]/(hydrogen_demand*day_pr_year)
                #total_cost.loc[i + '+' + j, 'sum'] = energy_cost_year.loc[i] + \
                #                                    tariff.loc[i + '+' + j]
        else:
            total_cost_w_tartiff[prefix].loc[i, 'investment'] = \
                                            inv_cost_elec/(hydrogen_demand*day_pr_year)
            total_cost_w_tartiff[prefix].loc[i, 'energy'] = \
                            energy_cost_year.loc[i]/(hydrogen_demand*day_pr_year)
                            
            


fig = plt.figure('Hydrogen Cost Breakdown')
ax_breakdown = plt.gca()



for prefix in prefix_list:                                        
    total_cost[prefix].plot(kind = 'bar', stacked = True, ax = ax_breakdown)
    ax_breakdown.set_ylabel('Hydrogen Cost [USD/kg]')

const_hydrogen_cost = pd.DataFrame()
#fig = plt.figure('Hydrogen Cost')
#ax_h2_cost = fig.gca() 
for prefix in prefix_list:
    const_hydrogen_cost[prefix] = total_cost[prefix].sum(axis = 1)
    #const_hydrogen_cost[prefix].plot(marker = '_', color = 'r',
    #                                   linewidth = 0.0,  ax = ax_h2_cost)#,
                             #xlim = (-1.0,len(const_hydrogen_cost[prefix])), ax = ax_h2_cost)
#ax_h2_cost.set_xticks(np.arange(len(const_hydrogen_cost[prefix].index)))
#ax_h2_cost.set_xticklabels(const_hydrogen_cost[prefix].index, rotation = 90)






