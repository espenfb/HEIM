# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 13:23:15 2018

@author: espenfb
"""

import detInvModel as dim
import datetime

time_data = {
'start_date' : datetime.datetime(2015,1,1),
'end_date' : datetime.datetime(2016,1,1),
'ref_date' : datetime.datetime(2015,1,1)}
dirs = {
'data_dir' : "Data\\",
'ctrl_data_file' : 'ctrl_data.csv',
'res_dir' : 'Result\\'}
obj = dim.deterministicModel(time_data, dirs)

obj.solve()

#obj.printModel()

obj.processResults()

obj.printRes()



