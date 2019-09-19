# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:55:31 2019

@author: espenfb
"""

import metaModel as mm
import pandas as pd
import numpy as np

time_data = {
'start_date': pd.Timestamp(year = 2015, month = 1, day = 1),
'end_date': pd.Timestamp(year = 2015, month = 12, day = 1),
'ref_date': pd.Timestamp(year = 2015, month = 1, day = 1)}
dirs = {
'data_dir' : "Data\\",
'ctrl_data_file' : 'ctrl_data.csv',
'res_dir' : 'Meta_results_newload\\'}

obj = mm.metaModel(time_data, dirs)


#param_type = 'inv_cost'
#param = 'H2_Storage'
#param_range = np.arange(0.23,0.28,0.005)

#obj.runMetaModel(param_type, param, param_range, orientation = 'row',
#                 key_col = 'Type', col_value = 'Cost')

param_type = 'parameters'
param = 'CO2_cost'
param_range = np.arange(0.03,0.29,0.03)


obj.runMetaModel(param_type, param, param_range)
