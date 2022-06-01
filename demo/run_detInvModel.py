# -*- coding: utf-8 -*-
# %% [markdown]
# Created on Thu Dec 20 13:23:15 2018
#
# @author: espenfb

# %%
import os
import sys
from pathlib import Path

filepath = os.path.abspath('')
sys.path.append(str(Path(filepath).parent / "src"))

import detInvModel as dim
import pandas as pd
import savedRes as sr

time_data = {
'start_date': pd.Timestamp(year = 2015, month = 1, day = 1),
'end_date': pd.Timestamp(year = 2016, month = 1, day = 1),
'ref_date': pd.Timestamp(year = 2015, month = 1, day = 1)}
dirs = {
'data_dir' : "demo\\data\\",
'ctrl_data_file' : 'ctrl_data.csv',
'res_dir' : 'results\\'}


# %%
obj = dim.deterministicModel(time_data, dirs)

# %%
obj.buildModel()

# %%
obj.solve()

# %%
obj.processResults()

obj.saveRes(dirs['res_dir'])

res = sr.savedRes(dirs['res_dir'], data = obj.data)

