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

inst_cap = pd.read_csv('..\\..\\production_capacity\\Installed_cap.csv',
                       index_col = 0)


