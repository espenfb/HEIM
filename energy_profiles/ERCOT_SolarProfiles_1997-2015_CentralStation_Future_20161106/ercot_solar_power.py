# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:46:34 2019

@author: espenfb
"""
import pandas as pd
import numpy as np

filename = '20161106_ERCOT_SOLAR_central-single_NetMW.CSV'

solar = pd.read_csv(filename, header = 0, index_col = 0, parse_dates = [[0,1]])

solar_id_cap = pd.DataFrame(columns = ['ID'])
for n, i in enumerate(solar.columns):
    w_id = int(i[5:10])
    solar_id_cap.loc[n,'ID'] = w_id

solar.columns = pd.Index(solar_id_cap['ID'], dtype = 'int')

solar = solar[solar.index.year == 2015]

solar_max = solar.max().to_frame(name = 'Cap')
solar_max.reset_index(level=0, inplace=True)

solar_cite_id = pd.read_csv('..\\solar_cite_id.csv',
                           skipinitialspace = True)

solar_cite_id.rename(columns = {'Primary' : 'ID'}, inplace = True)
solar_cite_id = pd.merge(solar_cite_id, solar_max, how='inner', on = ['ID'])

county_bus = pd.read_csv('..\\..\\grid\\bus_county.csv')

solar_bus = pd.merge(solar_cite_id, county_bus, how='inner', on = ['County'])

for i in solar_bus.index:
    row = solar_bus.loc[i]
    if not np.isnan(row.Secondary):
        row.ID = row.Secondary
        solar_bus = solar_bus.append(row)
solar_bus.drop(labels = ['Secondary'], axis = 1, inplace = True)

bus = solar_bus.groupby(['Bus','ID']).agg({'Cap':'sum'})

bus_tot = bus.sum(level = 0)

bus_tot.plot.bar()

solar_profiles = pd.DataFrame(columns = bus.index.levels[0])
for i in bus.index.levels[0]:
    solar_profile = pd.Series()
    for j in bus.loc[i].index:
        if solar_profile.empty:
            solar_profile = solar[j]
        else:
            solar_profile += solar[j]
    solar_profiles[i] = solar_profile

solar_profiles.plot()

solar_cap = solar_profiles.max().to_frame('Pot cap')
solar_cap.loc[:,'Inst cap'] = 0
solar_cap.to_csv('solar_cap.csv')

solar_profiles.to_csv('solar_profiles_bus.csv')

nrel_resource_pot = pd.DataFrame()
nrel_resource_pot.loc['Wind','Capacity'] = 1200 #GW 
nrel_resource_pot.loc['Wind','Energy'] = 4400 #TWh
nrel_resource_pot.loc['Solar','Capacity'] = 20400 #GW
nrel_resource_pot.loc['Solar','Energy'] = 41300 #TWh

installed_cap = pd.read_csv('..\\..\\production_capacity\\installed_cap_needs.csv', index_col = [0])
installed_cap_solar = installed_cap['Solar']

remaining_solar = nrel_resource_pot.loc['Solar','Capacity']*1E3 - installed_cap_solar.sum()
solar_profiles[7] = solar_profiles[8] 
solar_profiles_adj = (solar_profiles/solar_profiles.sum(axis = 1).max())*remaining_solar 
solar_cap_adj = solar_profiles_adj.max().to_frame('Pot cap')
solar_cap_adj['Inst cap'] = installed_cap_solar
solar_cap_adj.to_csv('solar_cap_adj.csv')

solar_profiles_adj_norm = (solar_profiles_adj/solar_profiles_adj.max())
solar_profiles_adj_norm[7] = solar_profiles_adj_norm[8]
solar_profiles_adj_norm.sort_index(axis =1, inplace = True)
solar_profiles_adj_norm.to_csv('solar_profiles_adj.csv')

inst_cap = pd.read_csv('..\\..\\production_capacity\\Installed_cap.csv',
                       index_col = 0)


