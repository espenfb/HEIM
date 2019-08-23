# -*- coding: utf-8 -*-
"""
Created on Mon May  6 20:57:59 2019

@author: espenfb
"""

import pandas as pd


kW2MW = 0.001
GWh2MWh = 1000


on_cost = pd.read_csv('cost_of_new_capacity.csv', index_col = 0) # 2015$/kW

fin_param = pd.read_csv('finalcial_parameters_new_capacity.csv', index_col = 0,
                        skipinitialspace = True)


irr = 0.06

life = fin_param.loc['Book Life Years']

eps = irr/(1-(1+irr)**(-life))




ann_cost = (on_cost*eps).round(-1)


ac = ann_cost.loc[2031].dropna() # 2015$/(kW*yr)

ac.index.name = 'Type'
ac = ac.to_frame('Cost')


ac.rename(index = {'CC': 'Gas', 'Solar PV': 'Solar'}, inplace = True)
ac.drop(['CT','IGCC'], inplace = True)

ac = ac/kW2MW# 2015$/(MW*yr)


ac.to_csv('investment_cost.csv')





####
#New data

eps = irr/(1-(1+irr)**(-cost['Lifetime']))

cost = pd.read_csv('new_parameters_plants.csv', index_col = 0,
                   skipinitialspace = True) # 2015$/kW


fuel_cost_btu = pd.read_csv('fuel_cost.csv', header = [0], index_col = 0,
                            skipinitialspace = True)

cost['Fuel cost [2018$/MWh]'] = cost['Heat rate [Btu/kWh]']*cost['Fuel cost [2018$/MMBtu]'].dropna()/GWh2MWh

cost['Tot var cost [2018$/MWh]'] = (cost['Fuel cost [2018$/MWh]']  + cost['Var cost [2018$/MWh]']).round(2)

cost['Tot inv cost [2018$/MWh]']  = ((cost['Inv cost [2018$/kW]']*eps + cost['Fixed cost [2018$/kW - yr]'])/kW2MW).round(-1)

cost['Tot var cost [2018$/MWh]'].to_csv('fuel_cost_2.csv', header = ['Cost'])

cost['Tot inv cost [2018$/MWh]'].to_csv('investment_cost_2.csv', header = ['Cost'])


cost['CO2 emissions[kg CO2/MWh]'] = cost['CO2 emissions[kg CO2/MMBtu]']*cost['Heat rate [Btu/kWh]']/GWh2MWh




cost['CO2 emissions[kg CO2/MWh]'].to_csv('emission_coeff.csv', header = ['Emission'])