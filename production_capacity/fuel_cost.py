# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:31:37 2019

@author: espenfb
"""

import pandas as pd

GWh2MWh = 1000

fuel_cost_btu = pd.read_csv('fuel_cost.csv', header = [0], index_col = 0,
                            skipinitialspace = True)

heat_rate = pd.read_csv('heat_rates.csv', header = [0], index_col = 0,
                            skipinitialspace = True)


fuel_cost_gwh = fuel_cost_btu['Baseline Price [$2015/MMBtu]']*heat_rate['Heat rate [Btu/kWh]'] # $/GWh
fuel_cost_mwh = fuel_cost_gwh.to_frame('fuel_cost')/GWh2MWh # $/MWh

fuel_cost_mwh.to_csv('fuel_cost_mwh.csv')