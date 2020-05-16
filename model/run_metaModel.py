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
'end_date': pd.Timestamp(year = 2015, month = 1, day = 10),
'ref_date': pd.Timestamp(year = 2015, month = 1, day = 1)}

dirs = {
'data_dir' : "Data\\",
'ctrl_data_file' : 'ctrl_data.csv',
#'res_dir' : 'NEEDS2\\Meta_results_100x\\'}
'res_dir' : 'Test\\'}

meta_data = {
'param': 'CO2_cost',
 'index': 'None',
 'kind': 'absolute',
 'range': np.arange(0.00,0.29,0.03)} 


#meta_data = {'param': 'Inv_cost',
#             'index': 'Wind',
#             'kind': 'relative',
#             'range': np.arange(-0.2,0.2,0.05)}

obj = mm.metaModel(time_data, dirs, meta_data)


obj.runMetaModel()

#obj.loadRes()
