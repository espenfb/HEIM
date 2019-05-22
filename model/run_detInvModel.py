# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 13:23:15 2018

@author: espenfb
"""

import detInvModel as dim
import datetime

dirs = {
'data_dir' : "Data\\",
'ctrl_data_file' : 'ctrl_data.csv',
'res_dir' : 'Result\\'}
obj = dim.deterministicModel(dirs)

obj.solve()

#obj.printModel()

obj.processResults()

obj.printRes()