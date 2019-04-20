# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:12:42 2019

@author: espenfb
"""

import pandas as pd

filename = 'natural_gas_price_henry_hub.xlsx'
sheet_name = 'Data 1'
skip_rows = [0,1]
header = 0

ng_price = pd.read_excel(filename,
                         sheet_name = sheet_name,
                          skiprows = skip_rows,
                          header = header,
                          index_col = 0,
                          parse_dates = [0])


