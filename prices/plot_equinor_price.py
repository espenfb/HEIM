# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 17:01:53 2019

@author: espenfb
"""

import pandas as pd

gas_price = pd.read_csv('Equinor/internal_natural_gas_price.csv', skiprows = [0],
                        index_col = 0, parse_dates = [0], skipinitialspace = True)

new_indx = []
for i in range(len(gas_price.index)):
    quarter = int(gas_price.Quarter.iloc[i][1])
    new_indx.append(gas_price.index[i].replace(month = quarter*3))
gas_price.index = new_indx

gas_price.plot()
