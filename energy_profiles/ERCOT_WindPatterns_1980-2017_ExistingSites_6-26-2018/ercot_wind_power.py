# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:46:34 2019

@author: espenfb
"""

import pandas as pd

filename = 'ERCOT_existing_1980-2017_20180625.CSV'

wind = pd.read_csv(filename, header = 0, index_col = 0, parse_dates = [[0,1]])
