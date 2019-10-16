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
'end_date': pd.Timestamp(year = 2015, month = 12, day = 31),
'ref_date': pd.Timestamp(year = 2015, month = 1, day = 1)}
dirs = {
'data_dir' : "Data\\",
'ctrl_data_file' : 'ctrl_data.csv',
'res_dir' : 'Meta_results_uc_2\\'}
meta_data = {'type': 'parameters',
             'param': 'CO2_cost',
             'range': np.arange(0.03,0.29,0.03)} 

obj = mm.metaModel(time_data, dirs, meta_data)


#obj.runMetaModel()

obj.loadRes()
