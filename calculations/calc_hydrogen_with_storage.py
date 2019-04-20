# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:16:04 2019

@author: espenfb
"""

import elecInvModel as eim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

## DATA ##
convert_timezone = True
year = 2017
#price_folder = '../Prices/'
price_folder = '../prices/Nordpool/'

#prefix = 'RTM'
#prefix = 'DAM'
#prefix_list = ['DAM', 'RTM']
prefix_list = ['SPOT','REG']
instances = {}
res = {}
res_zone = {}

NO_zone_map = {'NO1': 'Oslo', 'NO2':'Kr.sand', 'NO3':'Tr.heim', 'NO4':'Tromsø',
               'NO5': 'Bergen'}

for prefix in prefix_list:
    
    print('Price type: %s' % prefix)
    
    filename = prefix + '_' + str(year) + '.csv'
    file = price_folder + filename
    
    if prefix == 'SPOT':
        header = 0
    elif prefix == 'REG':
        header = [0,1]
    
    prices = pd.read_csv(file, index_col = 0, header = header, parse_dates = True)
    prices.index = pd.DatetimeIndex(prices.index)
    prices.dropna(inplace = True)
    prices = prices.tz_localize('Europe/Oslo', ambiguous = 'infer').tz_convert('utc')
    
    if prefix == 'SPOT':
        load_zone_col = prices.columns
    elif prefix == 'REG':
        load_zone_col = prices.columns.levels[0]
    
    lz_prices = prices[load_zone_col]
    
    hydrogen_param = {}
    density = 0.0899 #kg/Nm^3
    kW2MW = 0.001
    MW2kW = 1000
    
    elec_cost = 69469 # 2015EUR/MW
    storage_cost = 2.63225 # 2015EUR/Nm3
    EUR2USD = 1.0672 # USD/EUR
    inv_cost_elec = elec_cost*EUR2USD
    hydrogen_param['Elec_cost'] = inv_cost_elec
    inv_cost_storage = storage_cost*EUR2USD
    hydrogen_param['Storage_cost'] = inv_cost_storage
    hydrogen_param['Import_cost'] = 10000 #$/Nm^3
    
    conv_h2_nm3 = 4.66 #kW/Nm^3
    hydrogen_param['Elec_conv'] = conv_h2_nm3*kW2MW
    conv_pump =  4.79 # kW/Nm^3
    hydrogen_param['Storage_conv'] = conv_pump*kW2MW
    
    
    # Guessing prices are in $/MWh
    
    ## PARAMETERS ##
    hydrogen_demand = 50000 # kg/day
    day_pr_year = 365 # day/year
    month_in_year = 12 # month/year
    hours_pr_day = 24 # h/day
    
    hydrogen_demand_nm3 = (hydrogen_demand/density)/hours_pr_day
    
    hydrogen_param['Hydrogen_demand'] = hydrogen_demand_nm3
    hydrogen_param['Initial_storage'] = 0.5
    hydrogen_param['Elec_max'] = 200
    hydrogen_param['Storage_max'] = 1E6
    
    
    #zone = 'LZ_SOUTH'
    
    instances[prefix] = {}
    res_zone[prefix] = []
    res[prefix] = pd.DataFrame(columns = ['elec_cap','storage_cap'])
    
    for zone in tqdm(load_zone_col):
        
        
        if prefix == 'SPOT':
            price = lz_prices[zone].tolist()
            zone_name = zone
            if zone == 'SYS' or zone == 'Molde':
                continue
        elif prefix == 'REG':
            price = lz_prices[zone]['Down'].tolist()
            if zone in NO_zone_map.keys():
                zone_name = NO_zone_map[zone]
            else:
                zone_name = zone
        
        ## RUN MODEL ##
        model = eim.elecInvModel(price, hydrogen_param)
        
        model.solve(printOutput = False)
        
        instances[prefix][zone_name] = model.instance

        res_zone[prefix].append(zone_name)
        res[prefix].loc[res_zone[prefix][-1], 'elec_cap'] = \
            model.instance.elec_cap.get_values()[None]
        res[prefix].loc[res_zone[prefix][-1], 'storage_cap'] = \
            model.instance.storage_cap.get_values()[None]
        res[prefix].loc[res_zone[prefix][-1], 'hydrogen_price'] = \
            model.instance.obj.expr()/(day_pr_year*hydrogen_demand)
        
#markers = {'RTM': 'o',
#           'DAM': '+'}


for prefix in prefix_list:
    fig = plt.figure(prefix + ' - ' + 'Elec Cap')
    ax = fig.gca()
    res[prefix].elec_cap.plot(kind = 'bar', ax = ax)
    
    fig = plt.figure(prefix + ' - ' + 'Storage Cap')
    ax2 = fig.gca()
    res[prefix].storage_cap.plot(kind = 'bar', ax = ax2)
    
    
    fig = plt.figure(prefix + ' - ' + 'Storage Duration')
    ax3 = plt.gca()
    for zone in res_zone['REG']:
        val = list(instances[prefix][zone].storage_level.get_values().values())
        dur = np.sort(val)[::-1]
        ax3.plot(dur)
    

markers = {'CONST-SPOT': '*',
           'CONST-REG': '*',
           'FLEX-SPOT': 'o',
           'FLEX-REG': 'o'}
#color = {'CONST-SPOT': 'b',
#           'CONST-REG': 'b',
#           'FLEX-SPOT': 'g',
#           'FLEX-REG': 'g'}
    
hydrogen_price = pd.DataFrame()
hydrogen_price['CONST-SPOT'] = const_hydrogen_cost['SPOT']
hydrogen_price['CONST-REG'] = const_hydrogen_cost['REG']
hydrogen_price['FLEX-SPOT'] = res['SPOT']['hydrogen_price']
hydrogen_price['FLEX-REG'] = res['REG']['hydrogen_price']

    
for case in hydrogen_price.columns:    
    
    fig = plt.figure('Hydrogen Cost')
    ax = fig.gca()  
    hydrogen_price[case].plot(marker = markers[case], #color = color[case],
                             linewidth = 2.0, ax = ax)
                        # xlim = (-0.5,len(const_hydrogen_cost)-0.5))
ax.set_ylabel('Hydrogen Cost [USD/kg]')
ax.set_xticks(np.arange(len(hydrogen_price.index)))
ax.set_xticklabels(hydrogen_price.index, rotation = 90)
ax.legend(hydrogen_price.columns)

        