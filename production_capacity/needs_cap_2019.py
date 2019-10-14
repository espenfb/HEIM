# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 10:13:12 2019

@author: espenfb
"""
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np


cap_data_existing = pd.read_excel('..\\production_capacity\\needs_v6_may_2019_reference_case.xlsx',
                    sheet_name = 'NEEDS v6_active')

cap_data_hardwire = pd.read_excel('..\\production_capacity\\needs_v6_may_2019_reference_case.xlsx',
                    sheet_name = 'NEEDS v6_New_Capacity_Hardwired', skiprows = [0])

#cap_data_retireing = pd.read_excel('..\\production_capacity\\needs_v6_may_2019_reference_case.xlsx',
#                    sheet_name = 'NEEDS v6_Retired_Through2021')

columns = ['PlantType', 'Modeled Fuels', 'State Name','County','Capacity (MW)', 'Heat Rate (Btu/kWh)']

cap_data_all = pd.concat([cap_data_existing, cap_data_hardwire])

cap_data_all.PlantType.loc[cap_data_all['Combustion Turbine/IC Engine'] == 'IC Engine'] = 'ICE Gas'

cap_data_all.replace({'Mclennan': 'McLennan',
                      'Mcculloch': 'McCulloch'}, inplace = True)


cap = cap_data_all[columns]
cap = cap.loc[cap['State Name']=='Texas']
#cap.drop('State Name', axis = 1, inplace = True)


cap_county = cap.groupby(['County','PlantType']).agg({'Capacity (MW)': 'sum',
                      'Heat Rate (Btu/kWh)': 'mean'})

cap_type = cap.groupby(['PlantType']).agg({'Capacity (MW)': 'sum',
                      'Heat Rate (Btu/kWh)': 'mean'})
drop_types = cap_type.index[cap_type['Capacity (MW)'] < 100]

county2bus = pd.read_csv('bus_county.csv', index_col = 0)


cap['Bus'] = -1
for i in county2bus.Bus.unique():
    counties = county2bus.County[county2bus.Bus == i]
    county_index = cap.index[cap.County.isin(counties)]
    cap.loc[county_index, 'Bus'] = i
    
cap.replace({'O/G Steam': 'Combustion Turbine'}, inplace = True)  
cap_bus = cap.groupby(['Bus','PlantType']).agg({'Capacity (MW)': 'sum',
                      'Heat Rate (Btu/kWh)': 'mean'})
    
cap_bus.rename({'Onshore Wind': 'Wind',
                'Solar PV': 'Solar',
                'Combustion Turbine': 'CT Gas',
                'Combined Cycle': 'CC Gas',
                'Coal Steam':'Coal'}, inplace = True)
    
cap_bus_out = cap_bus[['Capacity (MW)']]
    
drop_index = cap_bus_out.index.isin(drop_types, level = 1)
keep_index = (drop_index == False)
cap_bus_out = cap_bus_out.loc[keep_index ]

cap_bus_out.drop(['Hydro', 'Energy Storage'],level = 1, inplace = True)
cap_bus_out = cap_bus_out.round(0)
cap_bus_out = cap_bus_out.unstack()['Capacity (MW)']
cap_bus_out['CCS Gas'] = np.NaN
cap_bus_out['CCS Coal'] = np.NaN
cap_bus_out.to_csv('installed_cap_needs.csv')