# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 13:23:15 2018

@author: espenfb
"""

import sys
if '..\\Inv_model\\' not in sys.path:
    sys.path.append( '..\\Inv_model\\')

import detInvModel as dim
import datetime

import numpy as np
import matplotlib.pyplot as plt


time_data = {
'start_date' : datetime.datetime(2015,1,1),
'end_date' : datetime.datetime(2016,1,1),
'ref_date' : datetime.datetime(2015,1,1)}
dirs = {
'data_dir' : "..\\Data\\",
'ctrl_data_file' : 'ctrl_data_ts_inv_2bus_det.csv',
'qf_dir' : '..\\Data\\Forecast_pandas_havoy\\',
'prod_dir' : '..\\Data\\Production\\',
'res_dir' : '..\\Result_inv_2bus_feb\\'}

obj = dim.deterministicModel(time_data, dirs)

obj.solve()

#obj.printModel()

obj.processResults()

obj.printRes()



