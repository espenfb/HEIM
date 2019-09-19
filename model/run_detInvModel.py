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
'end_date': pd.Timestamp(year = 2016, month = 1, day = 1),
'ref_date': pd.Timestamp(year = 2015, month = 1, day = 1)}
dirs = {
'data_dir' : "Data\\",
'ctrl_data_file' : 'ctrl_data.csv',
'res_dir' : 'Result\\'}
obj = dim.deterministicModel(time_data, dirs)

obj.solve()
#
#obj.printModel()
#
obj.processResults()

obj.saveRes('Result\\')

res = sr.savedRes('Result\\', data = obj.data)
