# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 13:23:15 2018

@author: espenfb
"""

import detInvModel_decomposed as dim
import pandas as pd
#import savedRes as sr

time_data = {
'start_date': pd.Timestamp(year = 2015, month = 1, day = 1),
'end_date': pd.Timestamp(year = 2015, month = 12, day = 29),
'ref_date': pd.Timestamp(year = 2015, month = 1, day = 1)}
dirs = {
'data_dir' : "Data\\",
'ctrl_data_file' : 'ctrl_data.csv',
'res_dir' : 'Result_decomposed_year\\'}
obj = dim.deterministicModel(time_data, dirs)

obj.run(maxItr = 100)

#obj.processResults()

#obj.saveRes(dirs['res_dir'])
#
#res = sr.savedRes(dirs['res_dir'], data = obj.data)