# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:46:34 2019

@author: espenfb
"""

import pandas as pd

filename = 'ERCOT_existing_1980-2017_20180625.CSV'

wind = pd.read_csv(filename, header = 0, index_col = 0, parse_dates = [[0,1]])

wind_id_cap = pd.DataFrame(columns = ['ID','CAP'])
for n, i in enumerate(wind.columns):
    w_id = int(i[5:10])
    w_cap = float(i[20:])
    wind_id_cap.loc[n,'ID'] = w_id
    wind_id_cap.loc[n,'CAP'] = w_cap
    



wind.columns = wind_id_cap['ID']

wind = wind[wind.index.year == 2017]


wind_keys = pd.read_csv('Existing_Wind_Site_ID_Key.csv',
                        skipinitialspace = True)


wind_cite_id = pd.read_csv('..\\wind_cite_id_2017.csv',
                           skipinitialspace = True)

wind_cite_id.rename(columns = {'SITE #' : 'ID'}, inplace = True)

wind_info = pd.merge(wind_id_cap,wind_cite_id, how='inner', on = ['ID'])

wind_county = wind_info.groupby(['ID','County']).agg({'Capacity (MW)':'sum'}).reset_index(level = ['County', 'ID'])

county_bus = pd.read_csv('..\\..\\grid\\county_bus.csv')

wind_bus = pd.merge(wind_county, county_bus, how='inner', on = ['County'])


bus = wind_bus.groupby(['Bus','ID']).agg({'Capacity (MW)':'sum'})


tot_bus = bus.sum(level = 0)

for c in bus.index.levels[0]:
    bus.loc[c, 'wind_share'] = (bus.loc[c]/tot_bus.loc[c])['Capacity (MW)'].tolist()
    

wind_profiles = pd.DataFrame(columns = bus.index.levels[0])
for i in bus.index.levels[0]:
    wind_profile = pd.Series()
    for j in bus.loc[i].index:
        if wind_profile.empty:
            wind_profile = bus.loc[i].loc[j]['wind_share']*wind[j]
        else:
            wind_profile += bus.loc[i].loc[j]['wind_share']*wind[j]
    wind_profiles[i] = wind_profile







