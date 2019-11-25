# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 13:23:15 2018

@author: espenfb
"""

import detInvModel as dim
import pandas as pd
import savedRes as sr

time_data = {
'start_date': pd.Timestamp(year = 2015, month = 1, day = 1),
'end_date': pd.Timestamp(year = 2015, month = 1, day = 31),
'ref_date': pd.Timestamp(year = 2015, month = 1, day = 1)}
dirs = {
'data_dir' : "Data\\",
'ctrl_data_file' : 'ctrl_data.csv',
'res_dir' : 'Result_test_uc\\'}
obj = dim.deterministicModel(time_data, dirs)

obj.buildModel()

obj.solve()
#
#obj.printModel()
#
obj.processResults()

obj.saveRes(dirs['res_dir'])

res = sr.savedRes(dirs['res_dir'], data = obj.data)

#
#  Matrix range     [2e-06, 4e+03]
#  Objective range  [1e+00, 2e+07]
#  Bounds range     [1e+00, 4e+03]
#  RHS range        [9e-05, 6e+06]
