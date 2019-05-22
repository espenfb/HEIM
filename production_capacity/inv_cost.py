# -*- coding: utf-8 -*-
"""
Created on Mon May  6 20:57:59 2019

@author: espenfb
"""

import pandas as pd


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

ac.to_csv('investment_cost.csv')